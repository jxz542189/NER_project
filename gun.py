import os
import gevent.monkey
gevent.monkey.patch_all()
import multiprocessing
debug = True
loglevel = 'debug'
bind = '0.0.0.0:8888'
pidfile = 'bert_blstm_crf/gun_log/gunicorn.pid'
logfile = 'bert_blstm_crf/gun_log/debug.log'
#daemon = True
threads = 2
workers = 3
worker_class = 'gunicorn.workers.ggevent.GeventWorker'

x_forwarded_for_header = 'X-FORWARDED-FOR'

