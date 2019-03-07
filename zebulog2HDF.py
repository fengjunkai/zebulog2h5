 
# -*- coding: utf-8 -*-
import os, time
from os import path
# from tables import *
import pandas as pd
import logging
from controls_bak import EVENTID_TYPES, CMD_PACKET
import numpy as np
# from multiprocessing import Process, Queue, Lock, Pool
from multiprocessing.dummy import threading, Lock, Queue
from progress.bar import ShadyBar as Bar

import getopt
import sys
import warnings
import tables

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

VERSION = 'zebulog2HDF v0.1'

class worker(threading.Thread):
    def __init__(self, inqueue, progress):
        self.inq = inqueue
        self.progress = progress

        super(worker,self).__init__()

    def run(self):
        logging.debug("Thread Start!")
 
        while True:
            (mode, store, lock, chunk, e) = self.inq.get()
            logging.debug('Thread ->> {}, {}'.format(mode, e))
            # self.outq.put("S")            
            try:
                # logging.debug(' ---> Start')
                t1 = time.time()
                if mode == 'LATENCY':
                    end_df = chunk.loc[:,['ts','eventID','timeCost']]
                    end_df['ts'] = chunk.loc[:,['ts']] + chunk.loc[:,['timeCost']]
                    end_df.eventID.replace(to_replace=e, value=EVENTID_TYPES[e] + '1', inplace=True)
                    # end_df.columns = ['ts','eventID']

                    start_df = chunk.loc[:,['ts','eventID','timeCost']]
                    start_df.eventID.replace(to_replace=e, value=EVENTID_TYPES[e] + '1', inplace=True)

                    new_df = pd.concat([start_df, end_df])
                    new_df.sort_values('ts', inplace=True)
                    new_df = new_df.set_index(['ts'], drop=True)
                    # new_df = new_df.loc[:,['timeCost', 'eventID']]
                    t2 = time.time()
                    lock.acquire(blocking=True)
                    t3 = time.time()
                    store.append(e, new_df, format='table')
                    store.flush()
                    lock.release()
                elif mode == 'PACKET':
                    # TODO need add stream ID info to text
                    end_df = chunk.loc[chunk['PnF'] == 1, ['ts','cmdType','timeCost','PnF']]
                    end_df = end_df.loc[end_df['timeCost'] > 0].assign(ts = end_df.ts + end_df.timeCost)

                    end_df = end_df.assign(y = EVENTID_TYPES[e] + '1')
                    end_df = end_df.assign(cmd_color = 'green')

                    start_df = chunk.loc[:,['ts','cmdType','timeCost','PnF']]
                    start_df = start_df.assign(y = EVENTID_TYPES[e])
                    start_df = start_df.assign(cmd_color = lambda x: np.where(x.PnF, 'green', 'red'))

                    # start_df = start_df[['ts','cmdType','timeCost','cmd_color','y']]
                    new_df = pd.concat([start_df, end_df], sort=False)
                    
                    for c in pd.unique(new_df.cmdType):
                        new_df.cmdType.replace(to_replace=e, value=CMD_PACKET[c], inplace=True)
                    
                    new_df.sort_values('ts', inplace=True)
                    new_df.set_index(['ts'], drop=True)

                    # new_df = new_df.loc[:,['timeCost','PnF','cmdType','cmd_color','y']]
                    # logging.debug(new_df)
                    t2 = time.time()
                    lock.acquire(blocking=True)
                    t3 = time.time()
                    
                    store.append(e, new_df, format='table')
                    store.flush()
                   
                    lock.release()

                self.progress.next(len(chunk))
                t4 = time.time()
                s1 = '{}: {}'.format(mode, e)
                s2 = ' ---> End {:.4}, HDF Store %: {:.2%}, Wait Lock %: {:.2%}'.format((t4 - t1),  (t4-t3)/(t4-t1), (t3-t2)/(t4-t1))
                logging.debug(s1+s2)
                # logging.debug(s2)
            except Exception as err:
                logging.error(err)
            else:
                self.inq.task_done()
                # self.outq.put('E')

def update_index(index, name, ts_min, ts_max, nrows, key):
    index['report'].append(name)
    index['ts_min'].append(ts_min)
    index['ts_max'].append(ts_max)
    index['nrows'].append(nrows)
    index['keys'].append(key)


if __name__ == '__main__':

    NTHREADS = 20
    chunksize = 500000
    
    z_mode = 'blosc'
    
    # evtsrc = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/event_report.log'
    # latsrc = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/latency_report.log'
    # packsrc = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/package_report.log'
    # streamsrc = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/stream_report.log'
    # index_file = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/index.h5'

    # evtfile = os.path.abspath('../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/event.h5')
    # latfile = os.path.abspath('../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/latency.h5')
    # packfile = os.path.abspath('../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/packet.h5')
    # streamfile = os.path.abspath('../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/stream.h5')
    # index_file = os.path.abspath('../qingge.sun_201902231012_resnet_inference_no_profiler_1.0/index.h5')
    # log_dir = '../qingge.sun_201902231012_resnet_inference_no_profiler_1.0'

    opts,args = getopt.getopt(sys.argv[1:],'-h-i:-o-v',['help','inputdir=','outputdir=','version'])
    for opt_name,opt_value in opts:
        if opt_name in ('-h','--help'):
            print('''
            zebulog2HDF.py -d REPORT_FOLDER
            ''')
            
        if opt_name in ('-v','--version'):
            print(VERSION)
            
        if opt_name in ('-i','--inputdir'):
            if opt_name not in ('-o','--outputdir'):
                log_dir = opt_value
            
                evtsrc = ''+log_dir+'/event_report.log'
                latsrc = ''+log_dir+'/latency_report.log'
                packsrc = ''+log_dir+'/package_report.log'
                streamsrc = ''+log_dir+'/stream_report.log'
                index_file = ''+log_dir+'/index.h5'

                evtfile = os.path.abspath(''+log_dir+'/event.h5')
                latfile = os.path.abspath(''+log_dir+'/latency.h5')
                packfile = os.path.abspath(''+log_dir+'/packet.h5')
                streamfile = os.path.abspath(''+log_dir+'/stream.h5')
                index_file = os.path.abspath(''+log_dir+'/index.h5')
        if opt_name in ('-o','--outputdir'):      
            if len(args)==2:
                # log_dir = args[0]
                hdf_dir = opt_value
                print(log_dir)
                print(hdf_dir)
                evtsrc = ''+log_dir+'/event_report.log'
                latsrc = ''+log_dir+'/latency_report.log'
                packsrc = ''+log_dir+'/package_report.log'
                streamsrc = ''+log_dir+'/stream_report.log'
                index_file = ''+hdf_dir+'/index.h5'

                evtfile = os.path.abspath(''+hdf_dir+'/event.h5')
                latfile = os.path.abspath(''+hdf_dir+'/latency.h5')
                packfile = os.path.abspath(''+hdf_dir+'/packet.h5')
                streamfile = os.path.abspath(''+hdf_dir+'/stream.h5')
                index_file = os.path.abspath(''+hdf_dir+'/index.h5')

            
    logging.basicConfig(level=logging.INFO,
                format='%(levelname)s %(asctime)s <0x%(process)x><0x%(thread)x> [line:%(lineno)d] %(funcName)s() %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',)
                # filename='myapp.log',
                # filemode='w')

    index = dict(
        report=[], ts_min=[], ts_max=[], nrows=[], keys=[]
    )

    on_off = [True,True,True,True,]

    reports = {'event':[evtsrc, True], 
                'latency':[latsrc, True], 
                'packet':[packsrc, True],
                'stream':[streamsrc, True]}
    print('Report Folder --> {}'.format(log_dir))
    print('    Avalible Target: {}'.format( str([k for k in reports]).replace('\'',''))) 
    # get total lines
    total_rows = 0
    for report in reports.values():
        if report[1]:
            count = 0
            f = open(report[0],'r',encoding='latin-1')
            for line in f: 
                count += 1
            total_rows += count
            f.close()
    
    progress = Bar('    Converting ', max=total_rows, suffix='%(percent).2f%% [ %(elapsed)ds ]')


    t0 = time.time()

    if reports['event'][1]:
        starttime = time.time()        
        store_event = pd.HDFStore(evtfile, complevel=1, complib=z_mode, mode='w')

        event_report = pd.read_csv(path.join(log_dir, "event_report.log").replace(os.sep, '/'),
                                    names=['ts', 'eventID'],
                                    index_col=False,
                                    comment='#',
                                    delim_whitespace=True,
                                    usecols=[0, 1],
                                    converters={"ts": lambda x: int(x, 16)},
                                    #dtype={'eventID':'category'}
                                    )
        event_report.sort_values(by='ts',inplace=True)
        event_report = event_report.set_index(['ts'], drop=True)

        progress.next(len(event_report))
        
        event_keys = pd.unique( event_report.eventID )

        update_index(index, 'event', event_report.index.min(), event_report.index.max() , len(event_report), ''.join(''.join(str(e)+' ') for e in event_keys))

        for e in event_keys:
            store_event.put(e, event_report.loc[event_report['eventID'] == e ], format='table', columns=[])

        # logging.debug(store_event)
        store_event.flush()
        store_event.close()
        logging.debug(time.time()-starttime)
        # index_df.assign(
        #     ''
        # )
   
    threads = []
    inqueue = Queue()

    # pool = Pool(NTHREADS)
    # start all threads
    for i in range(NTHREADS):
        t = worker(inqueue, progress)
        # thread = pool.apply_async(run_latency, [store, inqueue, outqueue,])
        t.start()
        threads.append(t)


###############################################################################
#  Latency Report
    if reports['latency'][1]:
        starttime = time.time()    
        store_latency_lock = Lock()
        mode = 'LATENCY'
        store_latency = pd.HDFStore(latfile, complevel=1, complib=z_mode, mode='w')

        latency_report = pd.read_csv(path.join(log_dir, "latency_report.log").replace(os.sep, '/'),
            names=['ts', 'eventID', 'timeCost'],
            index_col=False,
            comment='#',
            delim_whitespace=True,
            usecols=[0, 1, 2],
            converters={"ts": lambda x: int(x, 16),
                        "timeCost": lambda x: int(x, 16)},
            chunksize=chunksize,
            # iterator=True
                #dtype={'eventID':'category'}
            )
        logging.debug('Read chrunk!!')
        ts_min = 0
        ts_max = 0
        nrows = 0
        event_keys = []
        for lr_chunk in latency_report:
            lr_chunk.set_index(['eventID'], drop=False, inplace=True)
            ts_min = ts_min if ts_min < lr_chunk.ts.min() else lr_chunk.ts.min()
            ts_max = ts_max if ts_max > lr_chunk.ts.max() else lr_chunk.ts.max()
            nrows += len(lr_chunk)
            # chunk = lr_chunk.get_chunk(min(int(chunksize * (len(pd.unique( lr_chunk.read(chunksize).eventID )/15))), chunksize))
            for e in pd.unique( lr_chunk.eventID ):
                event_keys.append(e)
                while inqueue.full():
                    time.sleep(0.5)
                    pass
                inqueue.put((mode, store_latency, store_latency_lock, lr_chunk.loc[lr_chunk['eventID'] == e], e)) 
            logging.debug(inqueue.qsize())
        
        event_keys = pd.unique(  event_keys )

        update_index(index, 'latency', ts_min, ts_max, nrows, ''.join(''.join(str(e)+' ') for e in event_keys))

        logging.debug(time.time()-starttime)


###############################################################################
#  Packet Report
    if reports['packet'][1]:
        starttime = time.time()    
        store_packet_lock = Lock()
        mode = 'PACKET'
        store_packet = pd.HDFStore(packfile, complevel=1, complib=z_mode, mode='w')

        packet_report = pd.read_csv(path.join(log_dir, "package_report.log").replace(os.sep, '/'),
                                    names=['ts',            #0
                                            #'streamID',     #1
                                            'timeCost',     #2
                                            #'pkgCnt',       #3
                                            'cmdType',      #4
                                            #'stream_index', #5
                                            #'dmaType',      #6
                                            #'engID',        #7
                                            #'sNo',          #8
                                            'PnF',          #9 
                                            #'SnC'
                                            ],         #10
                                    index_col=False,
                                    comment='#',
                                    delim_whitespace=True,
                                    usecols=[0, 
                                            1, 
                                            #2, 
                                            #3, 
                                            4, 
                                            #5, 
                                            #6, 
                                            #7, 
                                            #8, 
                                            9, 
                                            #10
                                            ],
                                    converters={"ts": lambda x: int(x, 16),
                                                "timeCost": lambda x: int(x, 16),
                                                #"pkgCnt": lambda x: int(x, 16),
                                                #"stream_index": lambda x: int(x, 16),
                                                #"engID": lambda x: int(x, 16),
                                                #"sNo": lambda x: int(x, 16),
                                                #"streamID" : lambda x: int(x, 16),
                                                #"dmaType" : lambda x: int(x, 16),
                                                #   "streamID" : lambda x: int(x, 16),                                                                                            
                                                },
                                        dtype={ #'cmdType':'category',
                                                'PnF':'bool', 
                                                #'SnC':'bool'
                                                },
                                        chunksize=chunksize,
                                        # iterator=True
                                        )
        event_key  = 'CMD_PACKET'
        for pr in packet_report:
            ts_min = ts_min if ts_min < pr.ts.min() else pr.ts.min()
            ts_max = ts_max if ts_max > pr.ts.max() else pr.ts.max()
            nrows += len(pr)            
            while inqueue.full():
                time.sleep(0.5)
                pass
            inqueue.put((mode, store_packet, store_packet_lock, pr, event_key))         
        update_index(index, 'packet', ts_min, ts_max, nrows, event_key )

    inqueue.join()
    logging.debug('inqueue done!!')   

    try:
        store_latency.close()
    except:
        logging.warning('no latency report')

    try:
        store_packet.close()
    except:
        logging.warning('no packet report')

    logging.debug(time.time()-starttime)
 
###############################################################################
#  Stream Report
    if reports['stream'][1]:
        starttime = time.time()

        store_stream = pd.HDFStore(streamfile, complevel=1, complib=z_mode, mode='w')

        stream_report = pd.read_csv(path.join(log_dir, "stream_report.log").replace(os.sep, '/'),
                                    names=['ts', 'streamID', 'timeCost',
                                            'slotID', 'RBCnt', 'streamNum'],
                                    index_col=False,
                                    comment='#',
                                    delim_whitespace=True,
                                    usecols=[0, 1, 2, 3, 4, 5],
                                    converters={"ts": lambda x: int(x, 16),
                                                "timeCost": lambda x: int(x, 16),
                                                "slotID": lambda x: int(x, 16),
                                                "RBCnt": lambda x: int(x, 16),
                                                "streamNum": lambda x: int(x, 16)},
                                        )
        stream_report.sort_values(by='ts',inplace=True)
        stream_report = stream_report.set_index(['ts'], drop=True)

        event_key  = 'Stream_ID'
        # TODO add stream key to db
        store_stream.put(streamfile, stream_report, format='table' )
        # logging.debug("stream report empty status: {}".format(stream_report.empty))


        update_index(index, 'stream', stream_report.index.min(), stream_report.index.max() , len(stream_report), event_key)

        store_stream.close()
        logging.debug(time.time() - starttime)

        progress.index = total_rows
        progress.next(0)
        
    progress.finish()
    index_pd = pd.DataFrame(index)
    index_pd.to_hdf(index_file, key='index', mode='w')
    logging.debug(time.time() - t0)