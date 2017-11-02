import numpy as np
import re
import sys
from PIL import Image, ImageOps

def pfm_write(file, image, scale=1):
    """
    write pfm img
    :param file: 
    :param image: 
    :param scale: 
    :return: N/A
    """
    file = open(file, 'wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def pfm_read(path):
    """
    read a pfm img
    :param path: path to img
    :return: H x W x C numpy img (originally tuple of img and scale, but scale should be one in SFD so ignored)
    """
    with open(path, 'rb') as file:
        header = file.readline().rstrip().decode(encoding='utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encoding='utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip().decode(encoding='utf-8'))
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data  # , scale


def color_read(path):
    return Image.open(path).convert('RGB')