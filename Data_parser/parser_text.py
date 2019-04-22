import pandas as pd
import glob


def extract(filename): 
   with open(filename, errors='ignore') as file: 
        replics = file.read().split('\n\n') 
        return ['\n'.join(rep.split('\n')[2:]) for rep in replics] 


def main(outfile, ncache):
    films = glob.glob('films/*.txt')[6500:]
    print(len(films))
    df = pd.DataFrame(columns=['text'])
    n = len(films)
    text = []
    for i, film in enumerate(films):
        text.extend(extract(film))
        if i % ncache == 0:
            local_df = pd.DataFrame(text, columns=['text'])
            df = df.append(local_df, ignore_index=True)
            del local_df
            text = []
            size = (df['text'].memory_usage()+df.index.memory_usage())/1024**2
            print(f'{i} / {n}, size = {size:.2f}')
     
    local_df = pd.DataFrame(text, columns=['text'])
    df = df.append(local_df, ignore_index=True)
    del local_df
    del text
    print('save')
    df.to_csv(outfile)



if __name__ == '__main__':
    main('films_2_6500.csv', 300)
