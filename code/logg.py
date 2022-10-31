import time
import logging


localtime = time.strftime("%Y%m%d-%H:%M:%S",time.localtime())
filename = "info_result_"+localtime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(filename)
# 使用logger写入日志信息
logger.info("")
logger.debuge("")