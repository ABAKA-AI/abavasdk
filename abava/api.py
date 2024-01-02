# -*-coding:utf-8 -*-

import base64
import hashlib
import hmac
import json
from . import exception
from .utils import general


class API:
    def __init__(self, AccessKey, SecretKey):
        if AccessKey == '' or AccessKey is None:
            raise exception.AbavaUnauthorizedException
        elif SecretKey == '' or SecretKey is None:
            raise exception.AbavaUnauthorizedException
        self.AccessKey = AccessKey
        self.SecretKey = SecretKey

    def signature_auth_task(self, timestamp, task_id):
        data = {'ak': self.AccessKey, 'timestamp': timestamp, 'export_task_id': task_id}
        h = hmac.new(general.to_bytes(self.SecretKey), general.to_bytes(json.dumps(data, separators=(',', ':'))),
                     hashlib.sha256)
        return base64.b64encode(h.digest()).decode()
