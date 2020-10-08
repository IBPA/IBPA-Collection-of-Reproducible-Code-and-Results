from ftplib import FTP
import sys
import gzip

class FtpConst(object):

    FORMAT          = 'utf-8'

    MODE_BIN        = 'b'
    MODE_TEXT       = 't'

    COMM_RETRIEVE   = 'RETR'

    NEW_LINE        = '\n'
    SPACE           = ' '
    EMPTY_BIN       = b''

def download(host, filename, local, mode):
    """download a file on ftp server to the local in binary

    input
        host        : (str) server host
        filename    : (str) path of target file on server
        local       : (str) path target file downloaded locally
        mode        : (str) 'b' for binary, 't' for text on server

    output
        None"""

    # FTP configuration
    ftp = FTP(host)
    ftp.encoding = FtpConst.FORMAT
    ftp.login()

    # Start reading from FTP server
    content_list = []

    if mode == FtpConst.MODE_TEXT:

        # text mode
        ftp.retrlines(
            cmd         = FtpConst.SPACE.join([
                FtpConst.COMM_RETRIEVE,
                filename]),
            callback    = content_list.append)
        content = FtpConst.NEW_LINE.join(content_list)

    elif mode == FtpConst.MODE_BIN:

        # binar mode
        ftp.retrbinary(
            cmd         = FtpConst.SPACE.join([
                FtpConst.COMM_RETRIEVE,
                filename]),
            callback    = content_list.append)
        content_b   = FtpConst.EMPTY_BIN.join(content_list)
        content     = gzip.decompress(content_b).decode(FtpConst.FORMAT)

    else:
        raise ValueError("mode has to be either 't' (text) or 'b' (binary)")

    ftp.quit()

    # Write to local
    with open(local, 'wb') as f:
        if sys.version_info < (3, 0):
            f.write(bytes(content))
        else:
            f.write(bytes(content,'utf8'))