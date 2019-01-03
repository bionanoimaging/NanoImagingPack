# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:34:37 2018

@author: ckarras
"""
import collections;
import numpy;
import sys;
import re;
import math;
import io;
import binascii;

def isprintable(string):
    """Return if all characters in string are printable.

    >>> isprintable('abc')
    True
    >>> isprintable(b'\01')
    False

    """
    string = string.strip()
    if len(string) < 1:
        return True
    if sys.version_info[0] == 3:
        try:
            return string.isprintable()
        except Exception:
            pass
        try:
            return string.decode('utf-8').isprintable()
        except Exception:
            pass
    else:
        if string.isalnum():
            return True
        printable = ('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST'
                     'UVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c')
        return all(c in printable for c in string)

def asbool(value, true=(b'true', u'true'), false=(b'false', u'false')):
    """Return string as bool if possible, else raise TypeError.

    >>> asbool(b' False ')
    False

    """
    value = value.strip().lower()
    if value in true:  # might raise UnicodeWarning/BytesWarning
        return True
    if value in false:
        return False
    raise TypeError()

def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    for a, b in (('\r\n', '\n'), ('\r', '\n'), ('\n\n', '\n'),
                 ('\t', ' '), ('  ', ' ')):
        string = string.replace(a, b)
    if compact:
        for a, b in (('\n', ' '), ('[ ', '['),
                     ('  ', ' '), ('  ', ' '), ('  ', ' ')):
            string = string.replace(a, b)
    return string.strip()

def pformat_xml(xml):
    """Return pretty formatted XML."""
    try:
        import lxml.etree as etree  # delayed import
        if not isinstance(xml, bytes):
            xml = xml.encode('utf-8')
        xml = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(xml, pretty_print=True, xml_declaration=True,
                             encoding=xml.docinfo.encoding)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')

def pformat(arg, width=79, height=24, compact=True):
    """Return pretty formatted representation of object as string.

    Whitespace might be altered.

    """
    if height is None or height < 1:
        height = 1024
    if width is None or width < 1:
        width = 256

    npopt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=100, linewidth=width)

    if isinstance(arg, basestring):
        if arg[:5].lower() in ('<?xml', b'<?xml'):
            if isinstance(arg, bytes):
                arg = bytes2str(arg)
            if height == 1:
                arg = arg[:4*width]
            else:
                arg = pformat_xml(arg)
        elif isinstance(arg, bytes):
            if isprintable(arg):
                arg = bytes2str(arg)
                arg = clean_whitespace(arg)
            else:
                numpy.set_printoptions(**npopt)
                return hexdump(arg, width=width, height=height, modulo=1)
        arg = arg.rstrip()
    elif isinstance(arg, numpy.record):
        arg = arg.pprint()
    else:
        import pprint  # delayed import
        compact = {} if sys.version_info[0] == 2 else dict(compact=compact)
        arg = pprint.pformat(arg, width=width, **compact)

    numpy.set_printoptions(**npopt)

    if height == 1:
        arg = clean_whitespace(arg, compact=True)
        return arg[:width]

    argl = list(arg.splitlines())
    if len(argl) > height:
        arg = '\n'.join(argl[:height//2] + ['...'] + argl[-height//2:])
    return arg


def xml2dict(xml, sanitize=True, prefix=None):
    """Return XML as dict.

    >>> xml2dict('<?xml version="1.0" ?><root attr="name"><key>1</key></root>')
    {'root': {'key': 1, 'attr': 'name'}}

    """
    from xml.etree import cElementTree as etree  # delayed import

    at = tx = ''
    if prefix:
        at, tx = prefix

    def astype(value):
        # return value as int, float, bool, or str
        for t in (int, float, asbool):
            try:
                return t(value)
            except Exception:
                pass
        return value

    def etree2dict(t):
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        d = {key: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = collections.defaultdict(list)
            for dc in map(etree2dict, children):
                for k, v in dc.items():
                    dd[k].append(astype(v))
            d = {key: {k: astype(v[0]) if len(v) == 1 else astype(v)
                       for k, v in dd.items()}}
        if t.attrib:
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    return etree2dict(etree.fromstring(xml))

def hexdump(bytestr, width=75, height=24, snipat=-2, modulo=2, ellipsis='...'):
    """Return hexdump representation of byte string.

    >>> hexdump(binascii.unhexlify('49492a00080000000e00fe0004000100'))
    '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............'

    """
    size = len(bytestr)
    if size < 1 or width < 2 or height < 1:
        return ''
    if height == 1:
        addr = b''
        bytesperline = min(modulo * (((width - len(addr)) // 4) // modulo),
                           size)
        if bytesperline < 1:
            return ''
        nlines = 1
    else:
        addr = b'%%0%ix: ' % len(b'%x' % size)
        bytesperline = min(modulo * (((width - len(addr % 1)) // 4) // modulo),
                           size)
        if bytesperline < 1:
            return ''
        width = 3*bytesperline + len(addr % 1)
        nlines = (size - 1) // bytesperline + 1

    if snipat is None or snipat == 1:
        snipat = height
    elif 0 < abs(snipat) < 1:
        snipat = int(math.floor(height * snipat))
    if snipat < 0:
        snipat += height

    if height == 1 or nlines == 1:
        blocks = [(0, bytestr[:bytesperline])]
        addr = b''
        height = 1
        width = 3 * bytesperline
    elif height is None or nlines <= height:
        blocks = [(0, bytestr)]
    elif snipat <= 0:
        start = bytesperline * (nlines - height)
        blocks = [(start, bytestr[start:])]  # (start, None)
    elif snipat >= height or height < 3:
        end = bytesperline * height
        blocks = [(0, bytestr[:end])]  # (end, None)
    else:
        end1 = bytesperline * snipat
        end2 = bytesperline * (height - snipat - 1)
        blocks = [(0, bytestr[:end1]),
                  (size-end1-end2, None),
                  (size-end2, bytestr[size-end2:])]

    ellipsis = str2bytes(ellipsis)
    result = []
    for start, bytestr in blocks:
        if bytestr is None:
            result.append(ellipsis)  # 'skip %i bytes' % start)
            continue
        hexstr = binascii.hexlify(bytestr)
        strstr = re.sub(br'[^\x20-\x7f]', b'.', bytestr)
        for i in range(0, len(bytestr), bytesperline):
            h = hexstr[2*i:2*i+bytesperline*2]
            r = (addr % (i + start)) if height > 1 else addr
            r += b' '.join(h[i:i+2] for i in range(0, 2*bytesperline, 2))
            r += b' ' * (width - len(r))
            r += strstr[i:i+bytesperline]
            result.append(r)
    result = b'\n'.join(result)
    if sys.version_info[0] == 3:
        result = result.decode('ascii')
    return result


if sys.version_info[0] == 2:
    #inttypes = int, long  # noqa
    inttypes = int  # noqa
#    def print_(*args, **kwargs):
#        """Print function with flush support."""
#        flush = kwargs.pop('flush', False)
#        print(*args, **kwargs)
#        if flush:
#            sys.stdout.flush()

    def bytes2str(b, encoding=None, errors=None):
        """Return string from bytes."""
        return b

    def str2bytes(s, encoding=None):
        """Return bytes from string."""
        return s

    def byte2int(b):
        """Return value of byte as int."""
        return ord(b)

    class FileNotFoundError(IOError):
        pass

 # TiffFrame = TiffPage  # noqa
else:
    inttypes = int
    basestring = str, bytes
    unicode = str
    print_ = print

    def bytes2str(b, encoding=None, errors='strict'):
        """Return unicode string from encoded bytes."""
        if encoding is not None:
            return b.decode(encoding, errors)
        try:
            return b.decode('utf-8', errors)
        except UnicodeDecodeError:
            return b.decode('cp1252', errors)

    def str2bytes(s, encoding='cp1252'):
        """Return bytes from unicode string."""
        return s.encode(encoding)

    def byte2int(b):
        """Return value of byte as int."""
        return b