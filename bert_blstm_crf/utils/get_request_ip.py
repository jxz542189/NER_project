from flask import request
import traceback
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.txt')
logger = logging.getLogger(__name__)

def get_ip():
    try:
        ip = request.remote_addr
        return ip
    except Exception as e:
        logger.warning("ip achieve failed: " + traceback.format_exc())
