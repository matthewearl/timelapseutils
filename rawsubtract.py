#!/usr/bin/python

from __future__ import print_function

import argparse
import gc
import random
import sys

import numpy
import openraw
import PIL.Image

class UsageError(Exception):
    """
    The user provided bad arguments.    
    
    """
    pass

class NotSupportedError(Exception):
    """
    The given input is not supported yet. 

    """
    pass

def _raw_file_to_array(raw_file):
    """
    Convert the raw image data into a numpy array.

    The returned array is a 2x2 array of CFA data, with float values in the
    range 0 to 1.

    """
    d = raw_file.get_raw_data()

    if d.get_bits_per_channel() != 12:
        raise NotSupportedError("Only 12 bits per channel supported")

    max_val = float(1 << d.get_bits_per_channel())

    a = numpy.fromstring(d.get_data(),
                         dtype=numpy.uint16) / max_val
    a = a.astype(numpy.float32, copy=False)

    a = numpy.resize(a, (d.get_height(), d.get_width()))

    return a

def _rggb_debayer(a):
    """
    Convert a raw numpy array into a debayered image.

    """

    assert a.shape[0] % 2 == 0
    assert a.shape[1] % 2 == 0

    # Split a 2D array into 4 2D arrays, one for R, two for G, and one for
    # B.
    def split_into_sub_images(a):
        a = numpy.resize(a, (a.shape[0] / 2, 2, a.shape[1] / 2, 2))
        a = numpy.transpose(a, (1, 3, 0, 2))
        return a

    # Do the inverse of the above.
    def combine_sub_images(a):
        a = numpy.transpose(a, (2, 0, 3, 1))
        a = numpy.resize(a, (a.shape[0] * 2, a.shape[2] * 2))
        return a

    # Create some masks with the same resolution as the input array. red_mask
    # has ones for elements that are red filtered, and zeros everywhere else.
    # Similar for green_mask and blue_mask
    half_shape = (a.shape[0] / 2, a.shape[1] / 2)
    z = numpy.zeros(half_shape, dtype=numpy.float32)
    o = numpy.ones(half_shape, dtype=numpy.float32)
    red_mask = numpy.array(
        [ [ o, z ],
          [ z, z ]
        ])
    red_mask = combine_sub_images(red_mask)
    green_mask = numpy.array(
        [ [ z, o ],
          [ o, z ]
        ])
    green_mask = combine_sub_images(green_mask)
    blue_mask = numpy.array(
        [ [ z, z ],
          [ z, o ]
        ])
    blue_mask = combine_sub_images(blue_mask)
    
    # Create a red, green and blue channels by multiplying the input by each
    # mask.
    red_mask *= a
    green_mask *= a
    green_mask *= 0.5
    blue_mask *= a
    unconvolved = numpy.array( [ red_mask, green_mask, blue_mask ] )
    del red_mask, green_mask, blue_mask
    gc.collect()

    # Perform a basic 2x2 box blur on each channel.
    out = numpy.zeros(unconvolved.shape, dtype=numpy.float32)
    for chan in range(3):
        out[chan, :, :] += unconvolved[chan, :, :]
        out[chan, :-1, :] += unconvolved[chan, 1:, :]
        out[chan, :-1, :-1] += unconvolved[chan, 1:, 1:]
        out[chan, :, :-1] += unconvolved[chan, :, 1:]

    return out
    
def _load_image(file_name):
    """
    Load an image from a file.

    """
    # Load the image.
    im = PIL.Image.open(file_name)
    im_array = numpy.frombuffer(im.tostring(), dtype=numpy.uint8)
    im_array = im_array.astype(numpy.float32, copy=False)

    # Convert into the range 0..1
    im_array = numpy.array(im_array) / 255.0

    # Convert into a array indexed by channel, y, x
    im_array = im_array.reshape(im.size[1], im.size[0], 3)
    im_array = numpy.transpose(im_array, (2, 0, 1))

    return im_array
    
def _save_image(im_array, file_name):
    """
    Save an image as a file.

    The input image as a (3, height, width) array, with values in the range
    0..1. The first axis corresponds with the red, green and blue channels.

    """
    # Take a copy so we don't mutate the input.
    #im_array = im_array.copy()

    # PIL expects lines in bottom-to-top order
    for chan in range(3):
        im_array[chan, :, :] = numpy.flipud(im_array[chan, :, :])

    # Clamp values to 0..1
    numpy.clip(im_array, 0., 1.0, out=im_array)

    # Convert into a 1D array with values in the order expected by PIL.
    im_array = numpy.transpose(im_array, (1, 2, 0))
    dims = im_array.shape[1], im_array.shape[0]
    im_array = im_array.flatten()

    # Convert to bytes in the range 0..255
    im_array *= 255.
    im_array = numpy.uint8(im_array)

    # Save the image.
    im = PIL.Image.frombuffer("RGB", dims, numpy.getbuffer(im_array))
    im.save(file_name)

def _generate_channel_map(in_chan, out_chan, file=sys.stdout):
    assert in_chan.shape == out_chan.shape
    assert len(in_chan.shape) == 2

    for i in range(1000):
        r, c = (random.randrange(in_chan.shape[0]),
                random.randrange(in_chan.shape[1]))
        print("{},{},{},{}".format(
              c, r, 255. * in_chan[r, c], 255. * out_chan[r, c]),
              file=file)

def main(argv):
    def float_tuple(s):
        return tuple(float(x.strip()) for x in s.split(","))

    parser = argparse.ArgumentParser(description="Subtract 2 raw files")
    parser.add_argument("input_files", nargs="+", help="Files to subtract")
    parser.add_argument("--output", required=True, help="Output filename")
    parser.add_argument("--value-scale", type=float,
                        help="Post-process value scale")
    parser.add_argument("--component-scale", type=float_tuple,
                        help="Post-process per-component value scale")
    parser.add_argument("--rotate", type=int,
                        help="Rotate output image N times CCW")
    parser.add_argument("--value-offset", type=float,
                        help="Subtract this value from output pixel values")
    parser.add_argument("--compare-file", type=str,
                        help="Generate [rgb].csv files comparing the output "
                             "file's pixel values with the given file's. "
                             "Useful for determining white-balance settings "
                             "of an image.")

    args = parser.parse_args(argv) 

    # Debayer and subtract the input images.
    print("Loading and debayering")
    if len(args.input_files) == 1:
        f = openraw.RawFile.from_path(args.input_files[0])
        a = _raw_file_to_array(f)
        im_array = _rggb_debayer(a)
        del a
        del f
        gc.collect()

    elif len(args.input_files) == 2:
        f1 = openraw.RawFile.from_path(args.input_files[0])
        f2 = openraw.RawFile.from_path(args.input_files[1])
        a = _raw_file_to_array(f1)
        a2 = _raw_file_to_array(f2)
        a -= a2

        im_array = _rggb_debayer(a)
        del f1, f2, a2, a
        gc.collect()
    else:
        raise UsageError("More than 2 files passed")

    # Adjust the colours of the image if the user requested it.
    print("Adjusting colours")
    if args.value_offset is not None:
        # This could be done as a single subtraction, but it is split up to
        # reduce transient memory consumed by the array being subtracted.
        for chan in range(3):
            im_array[chan] -= (numpy.ones(im_array[chan].shape,
                                         dtype=numpy.uint8)
                                    * (args.value_offset * 256.))
    if args.component_scale is None:
        component_scale = 1.0, 1.0, 1.0
    else:
        component_scale = args.component_scale
    if args.value_scale is not None:
        component_scale = tuple(args.value_scale * x
            for x in component_scale)
    for chan in range(3):
        im_array[chan] *= component_scale[chan]

    # Rotate the image if the user requested it.
    if args.rotate is not None:
        print("Rotating")
        im_array = numpy.transpose(im_array, (1, 2, 0))
        im_array = numpy.rot90(im_array, args.rotate)
        im_array = numpy.transpose(im_array, (2, 0, 1))

    if args.compare_file is not None:
        print("Generating channel maps")
        balanced_im = _load_image(args.compare_file)
        for idx, name in enumerate("rgb"):
            f = open("{}.csv".format(name), "w")
            _generate_channel_map(
                im_array[idx, -balanced_im.shape[1]:, :balanced_im.shape[2]],
                balanced_im[idx],
                file=f)
            f.close()
        del balanced_im

    print("Collecting garbage")
    gc.collect()

    # Save the image.
    print("Saving")
    _save_image(im_array, args.output)

if __name__ == "__main__":
    main(sys.argv[1:])
