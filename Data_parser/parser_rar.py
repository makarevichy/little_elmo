# -*- coding: utf-8 -*-
import os
import patoolib
import glob
import shutil


def remove(error):
    def rem_dec(func):
        def wrapper(file, *args, **kwargs):
            try:
                result = func(file, *args, **kwargs)
                os.remove(file)
            except:
                print(error)
            else:
                return result
        return wrapper
    return rem_dec


@remove('error unpack')
def unpuck_rar(filerar, outdir):
    patoolib.extract_archive(filerar, outdir=outdir)


def extract_srt(folder, outdir):
    file = glob.glob(folder+'/*.srt')
    if not file:
        file = glob.glob(folder+'/*.txt')
    try:
        shutil.move(file[0], outdir)
    except:
        print('extract srt error')
    else:
        shutil.rmtree(folder, ignore_errors=True)


@remove('error convert to txt')
def convert(infilename, outfilename): 
    with open(infilename, 'rb') as infile: 
        with open(outfilename, 'w') as outfile: 
            binary_text = infile.read() 
            try: 
                text = binary_text.decode('cp1251') 
            except: 
                text = binary_text.decode() 
            outfile.write(text)   


def main():

    RARS = glob.glob('./rars/*.rar')
    
    for rar in RARS:
        unpuck_rar(rar, 'films')

    FOLDERS = list(os.walk('films'))[0][1]

    for folder in FOLDERS:
        extract_srt('films/'+folder, 'films')

    FILES = glob.glob('films/*.srt')
    for filename in FILES:
        convert(filename, filename[:-3]+'txt') 


if __name__ == '__main__':
    main()
