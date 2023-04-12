"""
A simple user interface for XMem
"""

import os
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import sys
from argparse import ArgumentParser

import torch

from model.network import XMem
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

from PyQt5.QtWidgets import QApplication
from inference.interact.gui import App
from inference.interact.resource_manager import ResourceManager

torch.set_grad_enabled(False)


if __name__ == '__main__':
    
    # Arguments parsing
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/XMem.pth')
    parser.add_argument('--s2m_model', default='saves/s2m.pth')
    parser.add_argument('--fbrs_model', default='saves/fbrs.pth')

    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=None)

    parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)
    
    parser.add_argument('--num_objects', type=int, default=1)

    # Long-memory options
    # Defaults. Some can be changed in the GUI.
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=10)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')
    args = parser.parse_args()

    config = vars(args)
    config['enable_long_term'] = True
    config['enable_long_term_count_usage'] = True

    with torch.cuda.amp.autocast(enabled=not args.no_amp):

        # Load our checkpoint
        network = XMem(config, args.model).cuda().eval()

        # Loads the S2M model
        if args.s2m_model is not None:
            s2m_saved = torch.load(args.s2m_model)
            s2m_model = S2M().cuda().eval()
            s2m_model.load_state_dict(s2m_saved)
        else:
            s2m_model = None

        s2m_controller = S2MController(s2m_model, args.num_objects, ignore_class=255)
        if args.fbrs_model is not None:
            fbrs_controller = FBRSController(args.fbrs_model)
        else:
            fbrs_controller = None

        # Manages most IO
        resource_manager = ResourceManager(config)

        app = QApplication(sys.argv)
        ex = App(network, resource_manager, s2m_controller, fbrs_controller, config)
        sys.exit(app.exec_())
