import logging
from datetime import date
import time
import os
# os.chdir("/tf/tool/TadGAN2")

LOGGER = logging.getLogger(__name__)

td = "-".join([str(date.today()),str(int(time.time()))])
td = str(date.today())
# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
os.makedirs('./log', exist_ok=True)
file_handler = logging.FileHandler('./log/{}.txt'.format(td))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)