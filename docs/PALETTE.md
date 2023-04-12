# Palette

> Some image formats, such as GIF or PNG, can use a palette, which is a table of (usually) 256 colors to allow for better compression. Basically, instead of representing each pixel with its full color triplet, which takes 24bits (plus eventual 8 more for transparency), they use a 8 bit index that represent the position inside the palette, and thus the color.
-- https://docs.geoserver.org/2.22.x/en/user/tutorials/palettedimage/palettedimage.html

So those mask files that look like color images are single-channel, `uint8` arrays under the hood. When `PIL` reads them, it (correctly) gives you a two-dimensional array (`opencv` does not work AFAIK). If what you get is instead of three-dimensional, `H*W*3` array, then your mask is not actually a paletted mask, but just a colored image. Reading and saving a paletted mask through `opencv` or MS Paint would destroy the palette.

Our code, when asked to generate multi-object segmentation (e.g., DAVIS 2017/YouTubeVOS), always reads and writes single-channel mask. If there is a palette in the input, we will use it in the output. The code does not care whether a palette is actually used -- we can read grayscale images just fine.

Importantly, we use `np.unique` to determine the number of objects in the mask. This would fail if:

1. Colored images, instead of paletted masks are used.
2. The masks have "smooth" edges, produced by feathering/downsizing/compression. For example, when you draw the mask in a painting software, make sure you set the brush hardness to maximum.
