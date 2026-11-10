import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from weatherfusion.rain_detection import RUN,train

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=3)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    def progress(message):
        print(message,flush=True)
        with (RUN/'progress.log').open('a',encoding='utf-8') as handle:
            handle.write(datetime.now(ZoneInfo('Europe/Rome')).strftime('%H:%M:%S')+' '+str(message)+'\n')
    train(args.epochs,progress)
