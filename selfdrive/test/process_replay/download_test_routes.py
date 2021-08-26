#!/usr/bin/env python3
import os
import requests

from selfdrive.test.process_replay.test_processes import segments

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

for car_brand, segment in segments:

  route_name, segment_num = segment.rsplit("--", 1)
  rlog_path = "%s/%s" % (route_name.replace("|", "/"), segment_num)
  rlog_fn = rlog_path + "/rlog.bz2"
  rlog_url = BASE_URL + rlog_fn

  r = requests.get(rlog_url)
  if r.status_code == 200:
    if not os.path.exists(rlog_path):
      os.makedirs(rlog_path)
    with open(rlog_fn, 'wb') as f:
      print("Downloading: " + rlog_url)
      f.write(r.content)
  else:
    print("Failed to download: " + rlog_url)
