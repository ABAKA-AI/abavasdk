# -*-coding:utf-8 -*-


class AbavaException(Exception):
    code = None

    def __init__(self, message, errcode=None):
        self.message = message

        if errcode:
            self.code = errcode

        if self.code:
            super().__init__(f'<Response [{self.code}]> {message}')
        else:
            super().__init__(f'<Response> {message}')


class AbavaParameterException(AbavaException):
    "函数参数缺失或有误"
    code = 400


class AbavaUnauthorizedException(AbavaException):
    "认证错误"
    code = 401


class AbavaInternetException(AbavaException):
    "网络服务异常"
    code = 402


class AbavaNoResourceException(AbavaException):
    "无资源"
    code = 404


class AbavaDrawTypeException(AbavaException):
    "drawType错误"
    code = 405


class AbavaCommonException(AbavaException):
    code = 406