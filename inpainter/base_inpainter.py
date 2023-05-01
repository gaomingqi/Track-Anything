import os
import glob
from PIL import Image

import torch
import yaml
import cv2
import importlib
import numpy as np
from tqdm import tqdm

from inpainter.util.tensor_util import resize_frames, resize_masks


class BaseInpainter:
	def __init__(self, E2FGVI_checkpoint, device) -> None:
		"""
		E2FGVI_checkpoint: checkpoint of inpainter (version hq, with multi-resolution support)
		"""
		net = importlib.import_module('inpainter.model.e2fgvi_hq')
		self.model = net.InpaintGenerator().to(device)
		self.model.load_state_dict(torch.load(E2FGVI_checkpoint, map_location=device))
		self.model.eval()
		self.device = device
		# load configurations
		with open("inpainter/config/config.yaml", 'r') as stream: 
			config = yaml.safe_load(stream) 
		self.neighbor_stride = config['neighbor_stride']
		self.num_ref = config['num_ref']
		self.step = config['step']
		# config for E2FGVI with splits
		self.num_subset_frames = config['num_subset_frames']
		self.num_external_ref = config['num_external_ref']

	# sample reference frames from the whole video
	def get_ref_index(self, f, neighbor_ids, length):
		ref_index = []
		if self.num_ref == -1:
			for i in range(0, length, self.step):
				if i not in neighbor_ids:
					ref_index.append(i)
		else:
			start_idx = max(0, f - self.step * (self.num_ref // 2))
			end_idx = min(length, f + self.step * (self.num_ref // 2))
			for i in range(start_idx, end_idx + 1, self.step):
				if i not in neighbor_ids:
					if len(ref_index) > self.num_ref:
						break
					ref_index.append(i)
		return ref_index

	def inpaint_efficient(self, frames, masks, num_tcb, num_tca, dilate_radius=15, ratio=1):
		"""
		Perform Inpainting for video subsets
		frames: numpy array, T, H, W, 3
		masks: numpy array, T, H, W
		num_tcb: constant, number of temporal context before, frames
		num_tca: constant, number of temporal context after, frames
		dilate_radius: radius when applying dilation on masks
		ratio: down-sample ratio

		Output:
		inpainted_frames: numpy array, T, H, W, 3
		"""
		assert frames.shape[:3] == masks.shape, 'different size between frames and masks'
		assert ratio > 0 and ratio <= 1, 'ratio must in (0, 1]'
		
		# --------------------
		# pre-processing
		# --------------------
		masks = masks.copy()
		masks = np.clip(masks, 0, 1)
		kernel = cv2.getStructuringElement(2, (dilate_radius, dilate_radius))
		masks = np.stack([cv2.dilate(mask, kernel) for mask in masks], 0)
		T, H, W = masks.shape
		masks = np.expand_dims(masks, axis=3)    # expand to T, H, W, 1
		# size: (w, h)
		if ratio == 1:
			size = None
			binary_masks = masks
		else:
			size = [int(W*ratio), int(H*ratio)]
			size = [si+1 if si%2>0 else si for si in size]  # only consider even values
			# shortest side should be larger than 50
			if min(size) < 50:
				ratio = 50. / min(H, W)
				size = [int(W*ratio), int(H*ratio)]
			binary_masks = resize_masks(masks, tuple(size))
			frames = resize_frames(frames, tuple(size))          # T, H, W, 3
		# frames and binary_masks are numpy arrays
		h, w = frames.shape[1:3]
		video_length = T - (num_tca + num_tcb)  # real video length
		# convert to tensor
		imgs = (torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().unsqueeze(0).float().div(255)) * 2 - 1
		masks = torch.from_numpy(binary_masks).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
		imgs, masks = imgs.to(self.device), masks.to(self.device)
		comp_frames = [None] * video_length
		tcb_imgs = None
		tca_imgs = None
		tcb_masks = None
		tca_masks = None
		# --------------------
		# end of pre-processing
		# --------------------

		# separate tc frames/masks from imgs and masks
		if num_tcb > 0:
			tcb_imgs = imgs[:, :num_tcb]
			tcb_masks = masks[:, :num_tcb]
		if num_tca > 0:
			tca_imgs = imgs[:, -num_tca:]
			tca_masks = masks[:, -num_tca:]
			end_idx = -num_tca
		else:
			end_idx = T

		imgs = imgs[:, num_tcb:end_idx]
		masks = masks[:, num_tcb:end_idx]
		binary_masks = binary_masks[num_tcb:end_idx]	# only neighbor area are involved
		frames = frames[num_tcb:end_idx]				# only neighbor area are involved

		for f in tqdm(range(0, video_length, self.neighbor_stride), desc='Inpainting image'):
			neighbor_ids = [
				i for i in range(max(0, f - self.neighbor_stride),
								min(video_length, f + self.neighbor_stride + 1))
			]
			ref_ids = self.get_ref_index(f, neighbor_ids, video_length)

			# selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
			# selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
			
			selected_imgs = imgs[:, neighbor_ids]
			selected_masks = masks[:, neighbor_ids]
			# pad before
			if tcb_imgs is not None:
				selected_imgs = torch.concat([selected_imgs, tcb_imgs], dim=1)
				selected_masks = torch.concat([selected_masks, tcb_masks], dim=1)
			# integrate ref frames
			selected_imgs = torch.concat([selected_imgs, imgs[:, ref_ids]], dim=1)
			selected_masks = torch.concat([selected_masks, masks[:, ref_ids]], dim=1)
			# pad after
			if tca_imgs is not None:
				selected_imgs = torch.concat([selected_imgs, tca_imgs], dim=1)
				selected_masks = torch.concat([selected_masks, tca_masks], dim=1)

			with torch.no_grad():
				masked_imgs = selected_imgs * (1 - selected_masks)
				mod_size_h = 60
				mod_size_w = 108
				h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
				w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
				masked_imgs = torch.cat(
					[masked_imgs, torch.flip(masked_imgs, [3])],
					3)[:, :, :, :h + h_pad, :]
				masked_imgs = torch.cat(
					[masked_imgs, torch.flip(masked_imgs, [4])],
					4)[:, :, :, :, :w + w_pad]
				pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
				pred_imgs = pred_imgs[:, :, :h, :w]
				pred_imgs = (pred_imgs + 1) / 2
				pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
				for i in range(len(neighbor_ids)):
					idx = neighbor_ids[i]
					img = pred_imgs[i].astype(np.uint8) * binary_masks[idx] + frames[idx] * (
							1 - binary_masks[idx])
					if comp_frames[idx] is None:
						comp_frames[idx] = img
					else:
						comp_frames[idx] = comp_frames[idx].astype(
							np.float32) * 0.5 + img.astype(np.float32) * 0.5
			torch.cuda.empty_cache()
		inpainted_frames = np.stack(comp_frames, 0)
		return inpainted_frames.astype(np.uint8)

	def inpaint(self, frames, masks, dilate_radius=15, ratio=1):
		"""
		Perform Inpainting for video subsets
		frames: numpy array, T, H, W, 3
		masks: numpy array, T, H, W
		dilate_radius: radius when applying dilation on masks
		ratio: down-sample ratio

		Output:
		inpainted_frames: numpy array, T, H, W, 3
		"""
		assert frames.shape[:3] == masks.shape, 'different size between frames and masks'
		assert ratio > 0 and ratio <= 1, 'ratio must in (0, 1]'
		
		# set num_subset_frames
		num_subset_frames = self.num_subset_frames
		# split frames into subsets
		video_length = len(frames)
		num_splits = video_length // num_subset_frames
		id_splits = [[i*num_subset_frames, (i+1)*num_subset_frames] for i in range(num_splits)]  # id splits
		
		if num_splits ==  0:
			id_splits = [[0, video_length]]
		
		# if remaining split > num_subset_frames/2, add a new split, else, append to the last split
		if video_length - id_splits[-1][-1] > num_subset_frames / 3:
			id_splits.append([num_splits*num_subset_frames, video_length])
		else:
			diff = video_length - id_splits[-1][-1]
			id_splits = [[ids[0]+diff, ids[1]+diff] for ids in id_splits]
			id_splits[0][0] = 0		# if OOM, let it happen at the begining :D

		# if appending, convert the appended split to the FIRST one, avoiding OOM at last

		# perform inpainting for each split
		inpainted_splits = []
		for id_split in id_splits:
			video_split = frames[id_split[0]:id_split[1]]
			mask_split = masks[id_split[0]:id_split[1]]

			# | id_before | ----- | id_split[0] | ----- | id_split[1] | ----- | id_after |
			# for each split, consider its temporal context [-context_range] frames and [context_range] frames
			id_before = max(0, id_split[0] - self.step * self.num_external_ref)
			try:
				tcb_frames = np.stack([frames[idb] for idb in range(id_before, (id_split[0]-self.step) + 1, self.step)], 0)
				tcb_masks = np.stack([masks[idb] for idb in range(id_before, (id_split[0]-self.step) + 1, self.step)], 0)
				num_tcb = len(tcb_frames)
			except:
				num_tcb = 0
			id_after = min(video_length, id_split[1] + self.step * self.num_external_ref + 1)
			try:
				tca_frames = np.stack([frames[ida] for ida in range(id_split[1]+self.step, id_after, self.step)], 0)
				tca_masks = np.stack([masks[ida] for ida in range(id_split[1]+self.step, id_after, self.step)], 0)
				num_tca = len(tca_frames)
			except:
				num_tca = 0

			# concatenate temporal context frames/masks with input frames/masks (for parallel pre-processing)
			if num_tcb > 0:
				video_split = np.concatenate([tcb_frames, video_split], 0)
				mask_split = np.concatenate([tcb_masks, mask_split], 0)
			if num_tca > 0:
				video_split = np.concatenate([video_split, tca_frames], 0)
				mask_split = np.concatenate([mask_split, tca_masks], 0)
			
			torch.cuda.empty_cache()
			# inpaint each split
			inpainted_splits.append(self.inpaint_efficient(video_split, mask_split, num_tcb, num_tca, dilate_radius, ratio))
			torch.cuda.empty_cache()
		inpainted_frames = np.concatenate(inpainted_splits, 0)

		return inpainted_frames.astype(np.uint8)

	def inpaint_ori(self, frames, masks, dilate_radius=15, ratio=1):
		"""
		frames: numpy array, T, H, W, 3
		masks: numpy array, T, H, W
		dilate_radius: radius when applying dilation on masks
		ratio: down-sample ratio

		Output:
		inpainted_frames: numpy array, T, H, W, 3
		"""
		assert frames.shape[:3] == masks.shape, 'different size between frames and masks'
		assert ratio > 0 and ratio <= 1, 'ratio must in (0, 1]'
		masks = masks.copy()
		masks = np.clip(masks, 0, 1)
		kernel = cv2.getStructuringElement(2, (dilate_radius, dilate_radius))
		masks = np.stack([cv2.dilate(mask, kernel) for mask in masks], 0)

		T, H, W = masks.shape
		masks = np.expand_dims(masks, axis=3)    # expand to T, H, W, 1
		# size: (w, h)
		if ratio == 1:
			size = None
			binary_masks = masks
		else:
			size = [int(W*ratio), int(H*ratio)]
			size = [si+1 if si%2>0 else si for si in size]  # only consider even values
			# shortest side should be larger than 50
			if min(size) < 50:
				ratio = 50. / min(H, W)
				size = [int(W*ratio), int(H*ratio)]

			size = [160, 120]
			binary_masks = resize_masks(masks, tuple(size))
			frames = resize_frames(frames, tuple(size))          # T, H, W, 3
		# frames and binary_masks are numpy arrays
		h, w = frames.shape[1:3]
		video_length = T

		# convert to tensor
		imgs = (torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().unsqueeze(0).float().div(255)) * 2 - 1
		masks = torch.from_numpy(binary_masks).permute(0, 3, 1, 2).contiguous().unsqueeze(0)

		imgs, masks = imgs.to(self.device), masks.to(self.device)
		comp_frames = [None] * video_length

		for f in tqdm(range(0, video_length, self.neighbor_stride), desc='Inpainting image'):
			neighbor_ids = [
				i for i in range(max(0, f - self.neighbor_stride),
								min(video_length, f + self.neighbor_stride + 1))
			]
			ref_ids = self.get_ref_index(f, neighbor_ids, video_length)
			selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
			selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
			with torch.no_grad():
				masked_imgs = selected_imgs * (1 - selected_masks)
				mod_size_h = 60
				mod_size_w = 108
				h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
				w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
				masked_imgs = torch.cat(
					[masked_imgs, torch.flip(masked_imgs, [3])],
					3)[:, :, :, :h + h_pad, :]
				masked_imgs = torch.cat(
					[masked_imgs, torch.flip(masked_imgs, [4])],
					4)[:, :, :, :, :w + w_pad]
				pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
				pred_imgs = pred_imgs[:, :, :h, :w]
				pred_imgs = (pred_imgs + 1) / 2
				pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
				for i in range(len(neighbor_ids)):
					idx = neighbor_ids[i]
					img = pred_imgs[i].astype(np.uint8) * binary_masks[idx] + frames[idx] * (
							1 - binary_masks[idx])
					if comp_frames[idx] is None:
						comp_frames[idx] = img
					else:
						comp_frames[idx] = comp_frames[idx].astype(
							np.float32) * 0.5 + img.astype(np.float32) * 0.5
			torch.cuda.empty_cache()
		inpainted_frames = np.stack(comp_frames, 0)
		return inpainted_frames.astype(np.uint8)


if __name__ == '__main__':

	# # davis-2017
	# frame_path = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/parkour', '*.jpg'))
	# frame_path.sort()
	# mask_path = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/Annotations/480p/parkour', "*.png"))
	# mask_path.sort()

	# long and large video
	mask_path = glob.glob(os.path.join('/ssd1/gaomingqi/test-sample13', '*.npy'))
	mask_path.sort()
	frames = np.load('/ssd1/gaomingqi/revenger.npy')
	save_path = '/ssd1/gaomingqi/results/inpainting/avengers_split'

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	masks = []
	for ti, mid in enumerate(mask_path):
		masks.append(np.load(mid, allow_pickle=True))
		if ti > 1122:
			break

	masks = np.stack(masks[:len(frames)], 0)

	# ----------------------------------------------
	# how to use
	# ----------------------------------------------
	# 1/3: set checkpoint and device
	checkpoint = '/ssd1/gaomingqi/checkpoints/E2FGVI-HQ-CVPR22.pth'
	device = 'cuda:4'
	# 2/3: initialise inpainter
	base_inpainter = BaseInpainter(checkpoint, device)
	# 3/3: inpainting (frames: numpy array, T, H, W, 3; masks: numpy array, T, H, W)
	# ratio: (0, 1], ratio for down sample, default value is 1
	inpainted_frames = base_inpainter.inpaint(frames[:300], masks[:300], ratio=0.6)   # numpy array, T, H, W, 3

	# save
	for ti, inpainted_frame in enumerate(inpainted_frames):
		frame = Image.fromarray(inpainted_frame).convert('RGB')
		frame.save(os.path.join(save_path, f'{ti:05d}.jpg'))

	torch.cuda.empty_cache()
	print('switch to ori')

	# inpainted_frames = base_inpainter.inpaint_ori(frames[:50], masks[:50], ratio=0.1)
	# save_path = '/ssd1/gaomingqi/results/inpainting/avengers'
	# # ----------------------------------------------
	# # end
	# # ----------------------------------------------
	# # save
	# for ti, inpainted_frame in enumerate(inpainted_frames):
	# 	frame = Image.fromarray(inpainted_frame).convert('RGB')
	# 	frame.save(os.path.join(save_path, f'{ti:05d}.jpg'))
