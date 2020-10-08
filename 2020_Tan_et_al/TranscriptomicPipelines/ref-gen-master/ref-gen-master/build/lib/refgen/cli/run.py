"""command line interface running script"""

import sys
from argparse import ArgumentParser

from refgen import Extractor

def parse_args(argv):
    parser = ArgumentParser(description = 'Command-line interface' + \
        ' for reference genome extractor')
    parser.add_argument(
        'term',
        action      = 'store',
        nargs       = '+',
        type        = str,
        help        = 'organism name needed for reference genome')
    parser.add_argument(
        '--update',
        action      = 'store_true',
        default     = False,
        help        = 're-download summary file if included')
    parser.add_argument(
        '--output',
        action      = 'store',
        nargs       = 1,
        type        = str,
        default     = 'output.csv',
        help        = 'path to store gff file')
    args = parser.parse_args(argv)
    return (' '.join(args.term), args.update, args.output)

def choose_refgen(extr, term):
    """retrieve qualified reference genomes candidates, and allow users to
    choose one to download, return the option provided by the user"""

    chosen_ftp_url = None

    while True:
        refgen_list = extr.find_refgen(term)

        if len(refgen_list) == 0: # if no refgen, re-enter a term
            print("No referece genome detected for your organisms.")
            print("Re-enter: ", end = '')
            term = input()
            continue
        else:
            print("{} reference genome detected.".format(len(refgen_list)))
            print("{:<2s} {:20s} {:100s}".format('Id', 'Assembly', 'Name'))
            for i in range(len(refgen_list)):
                print("{:<2d} {:20s} {:100s}".format(i + 1, refgen_list[i][0],
                    refgen_list[i][1]))

            if len(refgen_list) == 1: # if only one refgen, return ftp_url
                chosen_ftp_url = refgen_list[0][2]
            else: # if more than one, allow users to choose one
                while True:
                    print("Choose an id: ", end = '')
                    choice = input()

                    if not choice.isdigit() or int(choice) < 1 or \
                            int(choice) > len(refgen_list):
                        print("Invalid input. ", end = '')
                        continue
                    else:
                        chosen_ftp_url = refgen_list[int(choice) - 1][2]
                        break
            break
    return chosen_ftp_url

def download(extr, ftp_url, output):
    """download the specified reference genome using the ftp_url returned"""

    print("Downloading...")
    extr.extract(ftp_url, output)
    print("Finished!")

def main():

    term, update, output    = parse_args(sys.argv[1:])
    extr                    = Extractor(update)
    ftp_url                 = choose_refgen(extr, term)
    download(extr, ftp_url, output)

if __name__ == '__main__':
    main()