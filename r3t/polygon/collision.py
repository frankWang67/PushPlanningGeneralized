import sys

import math as python_lib_Math
import math as Math
import inspect as python_lib_Inspect
import sys as buddy_PythonSys
from threading import Timer as buddy_tools_Timer
import sys as python_lib_Sys
import builtins as python_lib_Builtins
import functools as python_lib_Functools
import re as python_lib_Re
import time as python_lib_Time
import timeit as python_lib_Timeit
import traceback as python_lib_Traceback
from datetime import datetime as python_lib_datetime_Datetime
from datetime import timezone as python_lib_datetime_Timezone
from io import StringIO as python_lib_io_StringIO
from threading import Semaphore as Lock
from threading import RLock as sys_thread__Mutex_NativeRLock
import threading


class _hx_AnonObject:
    _hx_disable_getattr = False
    def __init__(self, fields):
        self.__dict__ = fields
    def __repr__(self):
        return repr(self.__dict__)
    def __contains__(self, item):
        return item in self.__dict__
    def __getitem__(self, item):
        return self.__dict__[item]
    def __getattr__(self, name):
        if (self._hx_disable_getattr):
            raise AttributeError('field does not exist')
        else:
            return None
    def _hx_hasattr(self,field):
        self._hx_disable_getattr = True
        try:
            getattr(self, field)
            self._hx_disable_getattr = False
            return True
        except AttributeError:
            self._hx_disable_getattr = False
            return False



class Enum:
    _hx_class_name = "Enum"
    __slots__ = ("tag", "index", "params")
    _hx_fields = ["tag", "index", "params"]
    _hx_methods = ["__str__"]

    def __init__(self,tag,index,params):
        self.tag = tag
        self.index = index
        self.params = params

    def __str__(self):
        if (self.params is None):
            return self.tag
        else:
            return self.tag + '(' + (', '.join(str(v) for v in self.params)) + ')'

Enum._hx_class = Enum


class AsyncTools:
    _hx_class_name = "AsyncTools"
    __slots__ = ()
    _hx_statics = ["aMapLimit", "aMap", "aMapSeries", "aEachLimit", "aEach", "aEachSeries", "aFilterLimit", "aFilter", "aFilterSeries", "aForEachOfMapLimit"]

    @staticmethod
    def aMapLimit(iterable,limit,cb,done):
        def _hx_local_0(item,_,done):
            cb(item,done)
        AsyncTools.aForEachOfMapLimit(iterable,limit,_hx_local_0,done)

    @staticmethod
    def aMap(iterable,cb,done):
        AsyncTools.aMapLimit(iterable,0,cb,done)

    @staticmethod
    def aMapSeries(iterable,cb,done):
        AsyncTools.aMapLimit(iterable,1,cb,done)

    @staticmethod
    def aEachLimit(iterable,limit,cb,done):
        def _hx_local_1(item,done):
            def _hx_local_0(err = None):
                done(err,True)
            cb(item,_hx_local_0)
        def _hx_local_2(err,items):
            done(err)
        AsyncTools.aMapLimit(iterable,limit,_hx_local_1,_hx_local_2)

    @staticmethod
    def aEach(iterable,cb,done):
        AsyncTools.aEachLimit(iterable,0,cb,done)

    @staticmethod
    def aEachSeries(iterable,cb,done):
        AsyncTools.aEachLimit(iterable,1,cb,done)

    @staticmethod
    def aFilterLimit(iterable,limit,cb,done):
        def _hx_local_1(item,done):
            def _hx_local_0(err,keep):
                if (err is not None):
                    done(err,None)
                else:
                    done(None,(haxe_ds_Option.Some(item) if keep else haxe_ds_Option._hx_None))
            cb(item,_hx_local_0)
        def _hx_local_4(err,results):
            if (err is not None):
                done(err,None)
            else:
                def _hx_local_2(o):
                    return python_internal_ArrayImpl._get(list(o.params), 0)
                def _hx_local_3(o):
                    return (not Type.enumEq(o,haxe_ds_Option._hx_None))
                done(None,list(map(_hx_local_2,list(filter(_hx_local_3,results)))))
        AsyncTools.aMapLimit(iterable,limit,_hx_local_1,_hx_local_4)

    @staticmethod
    def aFilter(iterable,cb,done):
        AsyncTools.aFilterLimit(iterable,0,cb,done)

    @staticmethod
    def aFilterSeries(iterable,cb,done):
        AsyncTools.aFilterLimit(iterable,1,cb,done)

    @staticmethod
    def aForEachOfMapLimit(iterable,limit,cb,done):
        complete = haxe_ds_IntMap()
        it = HxOverrides.iterator(iterable)
        completed = False
        running = 0
        pos = 0
        def _hx_local_0():
            currentPos = pos
            _g = []
            _g1 = 0
            _g2 = currentPos
            while (_g1 < _g2):
                key = _g1
                _g1 = (_g1 + 1)
                x = complete.h.get(key,None)
                _g.append(x)
            output = _g
            return output
        completedItems = _hx_local_0
        next = None
        def _hx_local_5():
            nonlocal completed
            nonlocal pos
            nonlocal running
            if (not completed):
                if (not it.hasNext()):
                    if (running <= 0):
                        if (not completed):
                            completed = True
                            done(None,completedItems())
                else:
                    nextItem = it.next()
                    pos = (pos + 1)
                    currentPos = (pos - 1)
                    running = (running + 1)
                    def _hx_local_4(err,mapped):
                        nonlocal completed
                        nonlocal running
                        if (not completed):
                            if (err is not None):
                                if (not completed):
                                    completed = True
                                    done(err,completedItems())
                            else:
                                running = (running - 1)
                                complete.set(currentPos,mapped)
                                next()
                    cb(nextItem,currentPos,_hx_local_4)
                    if (not completed):
                        if ((limit == 0) or ((running < limit))):
                            next()
        next = _hx_local_5
        next()
AsyncTools._hx_class = AsyncTools


class Class: pass


class Date:
    _hx_class_name = "Date"
    __slots__ = ("date", "dateUTC")
    _hx_fields = ["date", "dateUTC"]
    _hx_statics = ["fromTime", "makeLocal", "fromString"]

    def __init__(self,year,month,day,hour,_hx_min,sec):
        self.dateUTC = None
        if (year < python_lib_datetime_Datetime.min.year):
            year = python_lib_datetime_Datetime.min.year
        if (day == 0):
            day = 1
        self.date = Date.makeLocal(python_lib_datetime_Datetime(year,(month + 1),day,hour,_hx_min,sec,0))
        self.dateUTC = self.date.astimezone(python_lib_datetime_Timezone.utc)

    @staticmethod
    def fromTime(t):
        d = Date(2000,0,1,0,0,0)
        d.date = Date.makeLocal(python_lib_datetime_Datetime.fromtimestamp((t / 1000.0)))
        d.dateUTC = d.date.astimezone(python_lib_datetime_Timezone.utc)
        return d

    @staticmethod
    def makeLocal(date):
        try:
            return date.astimezone()
        except BaseException as _g:
            None
            tzinfo = python_lib_datetime_Datetime.now(python_lib_datetime_Timezone.utc).astimezone().tzinfo
            return date.replace(**python__KwArgs_KwArgs_Impl_.fromT(_hx_AnonObject({'tzinfo': tzinfo})))

    @staticmethod
    def fromString(s):
        _g = len(s)
        if (_g == 8):
            k = s.split(":")
            return Date.fromTime((((Std.parseInt((k[0] if 0 < len(k) else None)) * 3600000.) + ((Std.parseInt((k[1] if 1 < len(k) else None)) * 60000.))) + ((Std.parseInt((k[2] if 2 < len(k) else None)) * 1000.))))
        elif (_g == 10):
            k = s.split("-")
            return Date(Std.parseInt((k[0] if 0 < len(k) else None)),(Std.parseInt((k[1] if 1 < len(k) else None)) - 1),Std.parseInt((k[2] if 2 < len(k) else None)),0,0,0)
        elif (_g == 19):
            k = s.split(" ")
            _this = (k[0] if 0 < len(k) else None)
            y = _this.split("-")
            _this = (k[1] if 1 < len(k) else None)
            t = _this.split(":")
            return Date(Std.parseInt((y[0] if 0 < len(y) else None)),(Std.parseInt((y[1] if 1 < len(y) else None)) - 1),Std.parseInt((y[2] if 2 < len(y) else None)),Std.parseInt((t[0] if 0 < len(t) else None)),Std.parseInt((t[1] if 1 < len(t) else None)),Std.parseInt((t[2] if 2 < len(t) else None)))
        else:
            raise haxe_Exception.thrown(("Invalid date format : " + ("null" if s is None else s)))

Date._hx_class = Date


class EReg:
    _hx_class_name = "EReg"
    __slots__ = ("pattern", "matchObj", "_hx_global")
    _hx_fields = ["pattern", "matchObj", "global"]

    def __init__(self,r,opt):
        self.matchObj = None
        self._hx_global = False
        options = 0
        _g = 0
        _g1 = len(opt)
        while (_g < _g1):
            i = _g
            _g = (_g + 1)
            c = (-1 if ((i >= len(opt))) else ord(opt[i]))
            if (c == 109):
                options = (options | python_lib_Re.M)
            if (c == 105):
                options = (options | python_lib_Re.I)
            if (c == 115):
                options = (options | python_lib_Re.S)
            if (c == 117):
                options = (options | python_lib_Re.U)
            if (c == 103):
                self._hx_global = True
        self.pattern = python_lib_Re.compile(r,options)

EReg._hx_class = EReg


class Lambda:
    _hx_class_name = "Lambda"
    __slots__ = ()
    _hx_statics = ["array", "exists", "iter", "filter", "empty"]

    @staticmethod
    def array(it):
        a = list()
        i = HxOverrides.iterator(it)
        while i.hasNext():
            i1 = i.next()
            a.append(i1)
        return a

    @staticmethod
    def exists(it,f):
        x = HxOverrides.iterator(it)
        while x.hasNext():
            x1 = x.next()
            if f(x1):
                return True
        return False

    @staticmethod
    def iter(it,f):
        x = HxOverrides.iterator(it)
        while x.hasNext():
            x1 = x.next()
            f(x1)

    @staticmethod
    def filter(it,f):
        _g = []
        x = HxOverrides.iterator(it)
        while x.hasNext():
            x1 = x.next()
            if f(x1):
                _g.append(x1)
        return _g

    @staticmethod
    def empty(it):
        return (not HxOverrides.iterator(it).hasNext())
Lambda._hx_class = Lambda


class Reflect:
    _hx_class_name = "Reflect"
    __slots__ = ()
    _hx_statics = ["field", "setProperty"]

    @staticmethod
    def field(o,field):
        return python_Boot.field(o,field)

    @staticmethod
    def setProperty(o,field,value):
        field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
        if isinstance(o,_hx_AnonObject):
            setattr(o,field1,value)
        elif hasattr(o,("set_" + ("null" if field1 is None else field1))):
            getattr(o,("set_" + ("null" if field1 is None else field1)))(value)
        else:
            setattr(o,field1,value)
Reflect._hx_class = Reflect


class Std:
    _hx_class_name = "Std"
    __slots__ = ()
    _hx_statics = ["isOfType", "string", "parseInt"]

    @staticmethod
    def isOfType(v,t):
        if ((v is None) and ((t is None))):
            return False
        if (t is None):
            return False
        if ((type(t) == type) and (t == Dynamic)):
            return (v is not None)
        isBool = isinstance(v,bool)
        if (((type(t) == type) and (t == Bool)) and isBool):
            return True
        if ((((not isBool) and (not ((type(t) == type) and (t == Bool)))) and ((type(t) == type) and (t == Int))) and isinstance(v,int)):
            return True
        vIsFloat = isinstance(v,float)
        tmp = None
        tmp1 = None
        if (((not isBool) and vIsFloat) and ((type(t) == type) and (t == Int))):
            f = v
            tmp1 = (((f != Math.POSITIVE_INFINITY) and ((f != Math.NEGATIVE_INFINITY))) and (not python_lib_Math.isnan(f)))
        else:
            tmp1 = False
        if tmp1:
            tmp1 = None
            try:
                tmp1 = int(v)
            except BaseException as _g:
                None
                tmp1 = None
            tmp = (v == tmp1)
        else:
            tmp = False
        if ((tmp and ((v <= 2147483647))) and ((v >= -2147483648))):
            return True
        if (((not isBool) and ((type(t) == type) and (t == Float))) and isinstance(v,(float, int))):
            return True
        if ((type(t) == type) and (t == str)):
            return isinstance(v,str)
        isEnumType = ((type(t) == type) and (t == Enum))
        if ((isEnumType and python_lib_Inspect.isclass(v)) and hasattr(v,"_hx_constructs")):
            return True
        if isEnumType:
            return False
        isClassType = ((type(t) == type) and (t == Class))
        if ((((isClassType and (not isinstance(v,Enum))) and python_lib_Inspect.isclass(v)) and hasattr(v,"_hx_class_name")) and (not hasattr(v,"_hx_constructs"))):
            return True
        if isClassType:
            return False
        tmp = None
        try:
            tmp = isinstance(v,t)
        except BaseException as _g:
            None
            tmp = False
        if tmp:
            return True
        if python_lib_Inspect.isclass(t):
            cls = t
            loop = None
            def _hx_local_1(intf):
                f = (intf._hx_interfaces if (hasattr(intf,"_hx_interfaces")) else [])
                if (f is not None):
                    _g = 0
                    while (_g < len(f)):
                        i = (f[_g] if _g >= 0 and _g < len(f) else None)
                        _g = (_g + 1)
                        if (i == cls):
                            return True
                        else:
                            l = loop(i)
                            if l:
                                return True
                    return False
                else:
                    return False
            loop = _hx_local_1
            currentClass = v.__class__
            result = False
            while (currentClass is not None):
                if loop(currentClass):
                    result = True
                    break
                currentClass = python_Boot.getSuperClass(currentClass)
            return result
        else:
            return False

    @staticmethod
    def string(s):
        return python_Boot.toString1(s,"")

    @staticmethod
    def parseInt(x):
        if (x is None):
            return None
        try:
            return int(x)
        except BaseException as _g:
            None
            base = 10
            _hx_len = len(x)
            foundCount = 0
            sign = 0
            firstDigitIndex = 0
            lastDigitIndex = -1
            previous = 0
            _g = 0
            _g1 = _hx_len
            while (_g < _g1):
                i = _g
                _g = (_g + 1)
                c = (-1 if ((i >= len(x))) else ord(x[i]))
                if (((c > 8) and ((c < 14))) or ((c == 32))):
                    if (foundCount > 0):
                        return None
                    continue
                else:
                    c1 = c
                    if (c1 == 43):
                        if (foundCount == 0):
                            sign = 1
                        elif (not (((48 <= c) and ((c <= 57))))):
                            if (not (((base == 16) and ((((97 <= c) and ((c <= 122))) or (((65 <= c) and ((c <= 90))))))))):
                                break
                    elif (c1 == 45):
                        if (foundCount == 0):
                            sign = -1
                        elif (not (((48 <= c) and ((c <= 57))))):
                            if (not (((base == 16) and ((((97 <= c) and ((c <= 122))) or (((65 <= c) and ((c <= 90))))))))):
                                break
                    elif (c1 == 48):
                        if (not (((foundCount == 0) or (((foundCount == 1) and ((sign != 0))))))):
                            if (not (((48 <= c) and ((c <= 57))))):
                                if (not (((base == 16) and ((((97 <= c) and ((c <= 122))) or (((65 <= c) and ((c <= 90))))))))):
                                    break
                    elif ((c1 == 120) or ((c1 == 88))):
                        if ((previous == 48) and ((((foundCount == 1) and ((sign == 0))) or (((foundCount == 2) and ((sign != 0))))))):
                            base = 16
                        elif (not (((48 <= c) and ((c <= 57))))):
                            if (not (((base == 16) and ((((97 <= c) and ((c <= 122))) or (((65 <= c) and ((c <= 90))))))))):
                                break
                    elif (not (((48 <= c) and ((c <= 57))))):
                        if (not (((base == 16) and ((((97 <= c) and ((c <= 122))) or (((65 <= c) and ((c <= 90))))))))):
                            break
                if (((foundCount == 0) and ((sign == 0))) or (((foundCount == 1) and ((sign != 0))))):
                    firstDigitIndex = i
                foundCount = (foundCount + 1)
                lastDigitIndex = i
                previous = c
            if (firstDigitIndex <= lastDigitIndex):
                digits = HxString.substring(x,firstDigitIndex,(lastDigitIndex + 1))
                try:
                    return (((-1 if ((sign == -1)) else 1)) * int(digits,base))
                except BaseException as _g:
                    return None
            return None
Std._hx_class = Std


class Float: pass


class Int: pass


class Bool: pass


class Dynamic: pass


class StringBuf:
    _hx_class_name = "StringBuf"
    __slots__ = ("b",)
    _hx_fields = ["b"]
    _hx_methods = ["get_length"]

    def __init__(self):
        self.b = python_lib_io_StringIO()

    def get_length(self):
        pos = self.b.tell()
        self.b.seek(0,2)
        _hx_len = self.b.tell()
        self.b.seek(pos,0)
        return _hx_len

StringBuf._hx_class = StringBuf


class StringTools:
    _hx_class_name = "StringTools"
    __slots__ = ()
    _hx_statics = ["lpad"]

    @staticmethod
    def lpad(s,c,l):
        if (len(c) <= 0):
            return s
        buf = StringBuf()
        l = (l - len(s))
        while (buf.get_length() < l):
            s1 = Std.string(c)
            buf.b.write(s1)
        s1 = Std.string(s)
        buf.b.write(s1)
        return buf.b.getvalue()
StringTools._hx_class = StringTools


class Sys:
    _hx_class_name = "Sys"
    __slots__ = ()
    _hx_statics = ["systemName"]

    @staticmethod
    def systemName():
        _g = python_lib_Sys.platform
        x = _g
        if x.startswith("linux"):
            return "Linux"
        else:
            _g1 = _g
            _hx_local_0 = len(_g1)
            if (_hx_local_0 == 5):
                if (_g1 == "win32"):
                    return "Windows"
                else:
                    raise haxe_Exception.thrown("not supported platform")
            elif (_hx_local_0 == 6):
                if (_g1 == "cygwin"):
                    return "Windows"
                elif (_g1 == "darwin"):
                    return "Mac"
                else:
                    raise haxe_Exception.thrown("not supported platform")
            else:
                raise haxe_Exception.thrown("not supported platform")
Sys._hx_class = Sys


class buddy_BuddySuite:
    _hx_class_name = "buddy.BuddySuite"
    __slots__ = ("suite", "currentSuite", "describeQueue", "timeoutMs", "fail", "pending")
    _hx_fields = ["suite", "currentSuite", "describeQueue", "timeoutMs", "fail", "pending"]
    _hx_methods = ["describe", "xdescribe", "before", "after", "beforeEach", "beforeAll", "afterEach", "afterAll", "it", "xit"]
    _hx_statics = ["useDefaultTrace"]

    def __init__(self):
        self.pending = None
        self.fail = None
        self.timeoutMs = 5000
        def _hx_local_0():
            self.currentSuite = buddy_TestSuite("")
            return self.currentSuite
        self.suite = _hx_local_0()
        self.describeQueue = list()

    def describe(self,description,spec,_hasInclude = None):
        if (_hasInclude is None):
            _hasInclude = False
        suite = buddy_TestSuite(description)
        self.currentSuite.specs.add(buddy_TestSpec.Describe(suite,_hasInclude))
        _this = self.describeQueue
        _this.append(_hx_AnonObject({'suite': suite, 'spec': spec}))

    def xdescribe(self,description,spec,_hasInclude = None):
        if (_hasInclude is None):
            _hasInclude = False

    def before(self,init):
        self.beforeEach(init)

    def after(self,init):
        self.afterEach(init)

    def beforeEach(self,init):
        self.currentSuite.beforeEach.add(init)

    def beforeAll(self,init):
        self.currentSuite.beforeAll.add(init)

    def afterEach(self,init):
        self.currentSuite.afterEach.add(init)

    def afterAll(self,init):
        self.currentSuite.afterAll.add(init)

    def it(self,desc,spec = None,_hasInclude = None,pos = None,time = None):
        if (_hasInclude is None):
            _hasInclude = False
        if (time is None):
            time = 0
        if (self.currentSuite == self.suite):
            raise haxe_Exception.thrown("Cannot use 'it' outside of a describe block.")
        self.currentSuite.specs.add(buddy_TestSpec.It(desc,spec,_hasInclude,pos,time))

    def xit(self,desc,spec = None,_hasInclude = None,pos = None,time = None):
        if (_hasInclude is None):
            _hasInclude = False
        if (time is None):
            time = 0
        if (self.currentSuite == self.suite):
            raise haxe_Exception.thrown("Cannot use 'it' outside of a describe block.")
        self.currentSuite.specs.add(buddy_TestSpec.It(desc,None,_hasInclude,pos,time))

buddy_BuddySuite._hx_class = buddy_BuddySuite


class TestHeadbutt2D(buddy_BuddySuite):
    _hx_class_name = "TestHeadbutt2D"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = []
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_BuddySuite


    def __init__(self):
        _gthis = self
        super().__init__()
        def _hx_local_16():
            def _hx_local_3():
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeB = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                verts = list()
                _g = 0
                _g1 = shapeA.vertices
                while (_g < len(_g1)):
                    a = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                    _g = (_g + 1)
                    _g2 = 0
                    _g3 = shapeB.vertices
                    while (_g2 < len(_g3)):
                        b = (_g3[_g2] if _g2 >= 0 and _g2 < len(_g3) else None)
                        _g2 = (_g2 + 1)
                        x = (a.x - b.x)
                        y = (a.y - b.y)
                        if (y is None):
                            y = 0
                        if (x is None):
                            x = 0
                        this1 = glm_Vec2Base()
                        this1.x = x
                        this1.y = y
                        verts.append(this1)
                md = headbutt_twod_shapes_Polygon(verts)
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 0
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = -1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                x = 1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec2Base()
                this5.x = x
                this5.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec2Base()
                this6.x = x
                this6.y = y
                x = 0
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec2Base()
                this7.x = x
                this7.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec2Base()
                this8.x = x
                this8.y = y
                dirs = [this1, this2, this3, this4, this5, this6, this7, this8]
                _g = 0
                while (_g < len(dirs)):
                    dir = (dirs[_g] if _g >= 0 and _g < len(dirs) else None)
                    _g = (_g + 1)
                    mdSupport = md.support(dir)
                    support = headbutt_twod_Headbutt.calculateSupport(shapeA,shapeB,dir)
                    buddy_ShouldFloat.should(support.x).be(mdSupport.x,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 43, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                    buddy_ShouldFloat.should(support.y).be(mdSupport.y,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 44, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_3)
            _gthis.it("should calculate Minkowski difference supports with polygons",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 17, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_4():
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeB = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = 0.5
                y = 0.5
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                shapeB.setTransform(this1,0,this2)
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeA,shapeB))
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 60, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_4)
            _gthis.it("should detect collisions for two polygons which overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 48, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_5():
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeB = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = 5
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                shapeB.setTransform(this1,0,this2)
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeA,shapeB))
                buddy_ShouldDynamic.should(result).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 75, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_5)
            _gthis.it("shouldn't detect collisions for two polygons which don't overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 63, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_6():
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeB = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = 0.5
                y = 0.5
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                shapeB.setTransform(this1,0,this2)
                resultA = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeA,shapeB))
                resultB = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeB,shapeA))
                buddy_ShouldDynamic.should(resultA).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 91, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldDynamic.should(resultB).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 92, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_6)
            _gthis.it("shouldn't matter what order the polygons go in",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 78, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_7():
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeA,shapeA))
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 101, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_7)
            _gthis.it("should detect collisions between a shape and itself",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 95, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_8():
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                lineA = headbutt_twod_shapes_Line(this1,this2)
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                lineB = headbutt_twod_shapes_Line(this1,this2)
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(lineA,lineA))
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 109, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_8)
            _gthis.it("should detect collisions between two lines",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 104, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_9():
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                shapeA = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                lineA = headbutt_twod_shapes_Line(this1,this2)
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(shapeA,lineA))
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 120, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_9)
            _gthis.it("should detect collisions between a line and a polygon",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 112, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_10():
                x = 0
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                circ = headbutt_twod_shapes_Circle(this1,1)
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                lineA = headbutt_twod_shapes_Line(this1,this2)
                x = -5
                y = -5
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = -4
                y = -4
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                lineB = headbutt_twod_shapes_Line(this1,this2)
                resultA = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(circ,lineA))
                resultB = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(circ,lineB))
                buddy_ShouldDynamic.should(resultA).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 131, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldDynamic.should(resultB).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 132, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_10)
            _gthis.it("should detect collisions between a line and a circle",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 123, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_11():
                x = 0.25
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                circ = headbutt_twod_shapes_Circle(this1,1)
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                x = -1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec2Base()
                this4.x = x
                this4.y = y
                square = headbutt_twod_shapes_Polygon([this1, this2, this3, this4])
                result = headbutt_twod__TestResult_TestResult_Impl_.toBool(headbutt_twod_Headbutt.test(circ,square))
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 143, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_11)
            _gthis.it("should detect collisions between a polygon and a circle",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 135, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_12():
                x = 2
                y = 2
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                squareA = headbutt_twod_shapes_Rectangle(this1)
                x = 2
                y = 2
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                squareB = headbutt_twod_shapes_Rectangle(this1)
                x = 1.5
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                squareB.setTransform(this1,0,this2)
                result = headbutt_twod_Headbutt.intersect(headbutt_twod_Headbutt.test(squareA,squareB))
                buddy_ShouldDynamic.should(result).get_not().be(None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 152, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(result.intersection.x).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 153, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(result.intersection.y).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 154, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_12)
            _gthis.it("should calculate the intersection of two rectangles",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 146, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_13():
                x = 0
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                circA = headbutt_twod_shapes_Circle(this1,1)
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                circB = headbutt_twod_shapes_Circle(this1,0.5)
                result = headbutt_twod_Headbutt.intersect(headbutt_twod_Headbutt.test(circA,circB))
                circA1 = circA.radius
                v = (Math.PI / 4)
                ix = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = ((5 * Math.PI) / 4)
                ix1 = ((circA1 * ix) - (((circB.radius * ((Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v)))) + circB.get_centre().x)))
                circA1 = circA.radius
                v = (Math.PI / 4)
                iy = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                v = ((5 * Math.PI) / 4)
                iy1 = ((circA1 * iy) - (((circB.radius * ((Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v)))) + circB.get_centre().y)))
                buddy_ShouldFloat.should(result.intersection.x).beCloseTo(ix1,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 166, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(result.intersection.y).beCloseTo(iy1,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 167, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_13)
            _gthis.it("should calculate the intersection of two circles",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 157, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_14():
                x = 2
                y = 2
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                square = headbutt_twod_shapes_Rectangle(this1)
                x = 0.5
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 3.5
                y = 3
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                line = headbutt_twod_shapes_Line(this1,this2)
                result = headbutt_twod_Headbutt.intersect(headbutt_twod_Headbutt.test(square,line))
                buddy_ShouldFloat.should(result.intersection.x).be(0.5,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 175, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(result.intersection.y).be(0,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 176, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_14)
            _gthis.it("should calculate the intersection of a line and square",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 170, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            def _hx_local_15():
                x = 2
                y = 2
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                squareA = headbutt_twod_shapes_Rectangle(this1)
                x = 2
                y = 2
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                squareB = headbutt_twod_shapes_Rectangle(this1)
                x = 2.1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                squareB.setTransform(this1,0,this2)
                hit = headbutt_twod_Headbutt.test(squareA,squareB).colliding
                buddy_ShouldDynamic.should(hit).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 185, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                x = 2.1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                tmp = (Math.PI / 4)
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                squareB.setTransform(this1,tmp,this2)
                hit = headbutt_twod_Headbutt.test(squareA,squareB).colliding
                buddy_ShouldDynamic.should(hit).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 189, 'className': "TestHeadbutt2D", 'methodName': "new"}))
                x = 2.1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1.5
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                squareB.setTransform(this1,0,this2)
                hit = headbutt_twod_Headbutt.test(squareA,squareB).colliding
                buddy_ShouldDynamic.should(hit).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 193, 'className': "TestHeadbutt2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_15)
            _gthis.it("should collide between a rotated square and a not",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt2D.hx", 'lineNumber': 179, 'className': "TestHeadbutt2D", 'methodName': "new"}))
        self.describe("Using Headbutt 2D",buddy_TestFunc.Sync(_hx_local_16))
TestHeadbutt2D._hx_class = TestHeadbutt2D


class TestHeadbutt3D(buddy_BuddySuite):
    _hx_class_name = "TestHeadbutt3D"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = []
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_BuddySuite


    def __init__(self):
        _gthis = self
        super().__init__()
        def _hx_local_11():
            def _hx_local_0():
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                a = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                x = -0.5
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -0.5
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -0.5
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -0.5
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 0.5
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 0.5
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 0.5
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 0.5
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                b = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                result = headbutt_threed_Headbutt.test(a,b)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 32, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_0)
            _gthis.it("should detect collisions for two polyhedrons which overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 17, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_1():
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sa = headbutt_threed_shapes_Sphere(this1,1)
                x = 0.5
                y = 0.5
                z = 0.5
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sb = headbutt_threed_shapes_Sphere(this1,1)
                result = headbutt_threed_Headbutt.test(sa,sb)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 40, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_1)
            _gthis.it("should detect collisions for two spheres which overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 35, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_2():
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                a = headbutt_threed_shapes_Polyhedron([this1])
                x = 1
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                a.setTransform(this1,dest,this2)
                m = a.transform
                x = 0
                y = 0
                z = 0
                w = 1
                if (w is None):
                    w = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec4Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this1.w = w
                vi = this1
                this1 = glm_Vec4Base()
                this1.x = 0
                this1.y = 0
                this1.z = 0
                this1.w = 0
                dest = this1
                x = vi.x
                y = vi.y
                z = vi.z
                w = vi.w
                dest.x = ((((m._00 * x) + ((m._10 * y))) + ((m._20 * z))) + ((m._30 * w)))
                dest.y = ((((m._01 * x) + ((m._11 * y))) + ((m._21 * z))) + ((m._31 * w)))
                dest.z = ((((m._02 * x) + ((m._12 * y))) + ((m._22 * z))) + ((m._32 * w)))
                dest.w = ((((m._03 * x) + ((m._13 * y))) + ((m._23 * z))) + ((m._33 * w)))
                vo = dest
                buddy_ShouldFloat.should(vo.x).be(1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 50, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(vo.y).be(0,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 51, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(vo.z).be(0,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 52, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(vo.w).be(1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 53, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                b = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                x = 5
                y = 5
                z = 5
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                b.setTransform(this1,dest,this2)
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                o = b.support(this1)
                buddy_ShouldFloat.should(o.x).be(6,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 64, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(6,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 65, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(6,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 66, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                o = b.support(this1)
                buddy_ShouldFloat.should(o.x).be(4,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 69, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(4,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 70, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(4,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 71, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                a = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                o = a.support(this1)
                buddy_ShouldFloat.should(o.x).be(1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 81, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 82, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 83, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                o = a.support(this1)
                buddy_ShouldFloat.should(o.x).be(-1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 86, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(-1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 87, 'className': "TestHeadbutt3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(o.x).be(-1,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 88, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_2)
            _gthis.it("should calculate supports properly",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 43, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_3():
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                a = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                b = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                result = headbutt_threed_Headbutt.test(a,b)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 106, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_3)
            _gthis.it("should detect collisions for two polyhedrons which overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 91, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_4():
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = 0
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                a = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4])
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = 0
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                b = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4])
                x = 5
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                b.setTransform(this1,dest,this2)
                result = headbutt_threed_Headbutt.test(a,b)
                buddy_ShouldDynamic.should(result).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 124, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_4)
            _gthis.it("shouldn't detect collisions for two polyhedrons which don't overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 109, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_5():
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sa = headbutt_threed_shapes_Sphere(this1,1)
                x = 1.75
                y = 1.75
                z = 1.75
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sb = headbutt_threed_shapes_Sphere(this1,1)
                result = headbutt_threed_Headbutt.test(sa,sb)
                buddy_ShouldDynamic.should(result).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 132, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_5)
            _gthis.it("should detect collisions for two spheres which overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 127, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_6():
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sa = headbutt_threed_shapes_Sphere(this1,1)
                x = 2
                y = 2
                z = 2
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sb = headbutt_threed_shapes_Sphere(this1,1)
                result = headbutt_threed_Headbutt.test(sa,sb)
                buddy_ShouldDynamic.should(result).be(False,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 140, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_6)
            _gthis.it("shouldn't detect collisions for two spheres which don't overlap",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 135, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_7():
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                s = headbutt_threed_shapes_Sphere(this1,1)
                result = headbutt_threed_Headbutt.test(s,s)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 146, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_7)
            _gthis.it("should detect collisions between a shape and itself",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 143, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_8():
                x = 0.5
                y = 0.5
                z = 0.5
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                s = headbutt_threed_shapes_Sphere(this1,1)
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = -1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = -1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = -1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this5 = glm_Vec3Base()
                this5.x = x
                this5.y = y
                this5.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this6 = glm_Vec3Base()
                this6.x = x
                this6.y = y
                this6.z = z
                x = 1
                y = 1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this7 = glm_Vec3Base()
                this7.x = x
                this7.y = y
                this7.z = z
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this8 = glm_Vec3Base()
                this8.x = x
                this8.y = y
                this8.z = z
                p = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4, this5, this6, this7, this8])
                result = headbutt_threed_Headbutt.test(s,p)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 158, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_8)
            _gthis.it("should detect collisions between a sphere and a polygon",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 149, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_9():
                this1 = glm_Vec3Base()
                this1.x = 0
                this1.y = 0
                this1.z = 0
                x = 5
                y = 5
                z = 5
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                line = headbutt_threed_shapes_Line(this1,this2)
                x = 3
                y = 3
                z = 3
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                sphere = headbutt_threed_shapes_Sphere(this1,1)
                result = headbutt_threed_Headbutt.test(line,sphere)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 165, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_9)
            _gthis.it("should detect collisions between a line and a sphere",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 161, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            def _hx_local_10():
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                boxA = headbutt_threed_shapes_Box(this1)
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                boxB = headbutt_threed_shapes_Box(this1)
                x = 0
                y = 0.5
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                boxB.setTransform(this1,dest,this2)
                result = headbutt_threed_Headbutt.test(boxA,boxB)
                buddy_ShouldDynamic.should(result).be(True,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 173, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_10)
            _gthis.it("should detect collisions between two boxes",tmp,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 168, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            _gthis.xit("should calculate the intersection of two spheres",None,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 176, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            _gthis.xit("should calculate the intersection of two polyhedrons",None,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 177, 'className': "TestHeadbutt3D", 'methodName': "new"}))
            _gthis.xit("should calculate the intersection of a polygon and a sphere",None,None,_hx_AnonObject({'fileName': "test/TestHeadbutt3D.hx", 'lineNumber': 178, 'className': "TestHeadbutt3D", 'methodName': "new"}))
        self.describe("Using Headbutt 3D",buddy_TestFunc.Sync(_hx_local_11))
TestHeadbutt3D._hx_class = TestHeadbutt3D


class TestMain:
    _hx_class_name = "TestMain"
    __slots__ = ()
    _hx_statics = ["main"]

    @staticmethod
    def main():
        reporter = buddy_reporting_ConsoleFileReporter(True)
        runner = buddy_SuitesRunner([TestShapes2D(), TestHeadbutt2D(), TestShapes3D(), TestHeadbutt3D()],reporter)
        runner.run()
TestMain._hx_class = TestMain


class TestShapes2D(buddy_BuddySuite):
    _hx_class_name = "TestShapes2D"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = []
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_BuddySuite


    def __init__(self):
        _gthis = self
        super().__init__()
        def _hx_local_5():
            p = None
            def _hx_local_0():
                nonlocal p
                x = -1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = -1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                x = 0
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec2Base()
                this3.x = x
                this3.y = y
                p = headbutt_twod_shapes_Polygon([this1, this2, this3])
            tmp = buddy_TestFunc.Sync(_hx_local_0)
            _gthis.beforeEach(tmp)
            def _hx_local_1():
                p1 = p
                x = 0
                y = 3
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                p1.setTransform(this1,0,this2)
                c = p.get_centre()
                buddy_ShouldFloat.should(c.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 26, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.y).beCloseTo(3,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 27, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_1)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 23, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_2():
                p1 = p
                x = 0
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 32, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 33, 'className': "TestShapes2D", 'methodName': "new"}))
                p1 = p
                x = -0.5
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 36, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 37, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_2)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 30, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_3():
                p1 = p
                x = 0
                y = 3
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                p1.setTransform(this1,0,this2)
                p1 = p
                x = 0
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 44, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 45, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_3)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 40, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_4():
                p1 = p
                x = 0
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                tmp = (Math.PI / 2)
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                p1.setTransform(this1,tmp,this2)
                p1 = p
                x = -1
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(-1,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 52, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 53, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_4)
            _gthis.it("should calculate support vertices when rotated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 48, 'className': "TestShapes2D", 'methodName': "new"}))
        self.describe("Using A Polygon",buddy_TestFunc.Sync(_hx_local_5))
        def _hx_local_11():
            r = None
            def _hx_local_6():
                nonlocal r
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                r = headbutt_twod_shapes_Rectangle(this1)
            tmp = buddy_TestFunc.Sync(_hx_local_6)
            _gthis.beforeEach(tmp)
            def _hx_local_7():
                r1 = r
                x = 0
                y = 3
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                r1.setTransform(this1,0,this2)
                c = r.get_centre()
                buddy_ShouldFloat.should(c.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 66, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.y).beCloseTo(3,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 67, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_7)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 63, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_8():
                r1 = r
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = r1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 72, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 73, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_8)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 70, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_9():
                r1 = r
                x = 0
                y = 3
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                r1.setTransform(this1,0,this2)
                r1 = r
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = r1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 80, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(3.5,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 81, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_9)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 76, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_10():
                r1 = r
                x = 0
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                tmp = (Math.PI / 4)
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec2Base()
                this2.x = x
                this2.y = y
                r1.setTransform(this1,tmp,this2)
                r1 = r
                x = 0
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = r1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0.0,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 88, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 89, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_10)
            _gthis.it("should calculate support vertices when rotated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 84, 'className': "TestShapes2D", 'methodName': "new"}))
        self.describe("Using A Rectangle",buddy_TestFunc.Sync(_hx_local_11))
        def _hx_local_16():
            c = None
            def _hx_local_12():
                nonlocal c
                x = 0
                y = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                c = headbutt_twod_shapes_Circle(this1,1)
            tmp = buddy_TestFunc.Sync(_hx_local_12)
            _gthis.beforeEach(tmp)
            def _hx_local_13():
                x = 4
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                c.position = this1
                c1 = c.get_centre()
                buddy_ShouldFloat.should(c1.x).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 102, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c1.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 103, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_13)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 99, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_14():
                c1 = c
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = c1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 108, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 109, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_14)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 106, 'className': "TestShapes2D", 'methodName': "new"}))
            def _hx_local_15():
                c.position.x = 3
                c1 = c
                x = 1
                y = 1
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                support = c1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(((1 / python_lib_Math.sqrt(2)) + 3),None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 116, 'className': "TestShapes2D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 117, 'className': "TestShapes2D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_15)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes2D.hx", 'lineNumber': 112, 'className': "TestShapes2D", 'methodName': "new"}))
        self.describe("Using A Circle",buddy_TestFunc.Sync(_hx_local_16))
TestShapes2D._hx_class = TestShapes2D


class TestShapes3D(buddy_BuddySuite):
    _hx_class_name = "TestShapes3D"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = []
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_BuddySuite


    def __init__(self):
        _gthis = self
        super().__init__()
        def _hx_local_5():
            p = None
            def _hx_local_0():
                nonlocal p
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = 1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                x = 1
                y = -1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this3 = glm_Vec3Base()
                this3.x = x
                this3.y = y
                this3.z = z
                x = 0
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this4 = glm_Vec3Base()
                this4.x = x
                this4.y = y
                this4.z = z
                p = headbutt_threed_shapes_Polyhedron([this1, this2, this3, this4])
            tmp = buddy_TestFunc.Sync(_hx_local_0)
            _gthis.beforeEach(tmp)
            def _hx_local_1():
                p1 = p
                x = 0
                y = 3
                z = 4
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                p1.setTransform(this1,dest,this2)
                c = p.get_centre()
                buddy_ShouldFloat.should(c.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 29, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.y).beCloseTo(3,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 30, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.z).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 31, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_1)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 26, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_2():
                p1 = p
                x = 0
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 36, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 37, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 38, 'className': "TestShapes3D", 'methodName': "new"}))
                p1 = p
                x = -0.5
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 41, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 42, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 43, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_2)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 34, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_3():
                p1 = p
                x = 0
                y = 3
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                p1.setTransform(this1,dest,this2)
                p1 = p
                x = 0
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 50, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 51, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 52, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_3)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 46, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_4():
                p1 = p
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = Math.PI
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                v = (x / 2)
                c1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = 0.
                c2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = 0.
                c3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = (x / 2)
                s1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                v = 0.
                s2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                v = 0.
                s3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                dest.x = (((s1 * c2) * c3) + (((c1 * s2) * s3)))
                dest.y = (((c1 * s2) * c3) - (((s1 * c2) * s3)))
                dest.z = (((c1 * c2) * s3) + (((s1 * s2) * c3)))
                dest.w = (((c1 * c2) * c3) - (((s1 * s2) * s3)))
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                p1.setTransform(this1,dest,this2)
                p1 = p
                x = 0
                y = -1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = p1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 59, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(-1,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 60, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 61, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_4)
            _gthis.it("should calculate support vertices when rotated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 55, 'className': "TestShapes3D", 'methodName': "new"}))
        self.describe("Using A Polyhedron",buddy_TestFunc.Sync(_hx_local_5))
        def _hx_local_11():
            b = None
            def _hx_local_6():
                nonlocal b
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                b = headbutt_threed_shapes_Box(this1)
            tmp = buddy_TestFunc.Sync(_hx_local_6)
            _gthis.beforeEach(tmp)
            def _hx_local_7():
                b1 = b
                x = 0
                y = 3
                z = 4
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                b1.setTransform(this1,dest,this2)
                c = b.get_centre()
                buddy_ShouldFloat.should(c.x).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 74, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.y).beCloseTo(3,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 75, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.z).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 76, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_7)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 71, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_8():
                b1 = b
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = b1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 81, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 82, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 83, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_8)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 79, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_9():
                b1 = b
                x = 0
                y = 3
                z = 4
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                dest.x = 0
                dest.y = 0
                dest.z = 0
                dest.w = 1
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                b1.setTransform(this1,dest,this2)
                b1 = b
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = b1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 90, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(3.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 91, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(4.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 92, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_9)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 86, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_10():
                b1 = b
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                x = Math.PI
                this2 = glm_QuatBase()
                this2.x = 0
                this2.y = 0
                this2.z = 0
                this2.w = 1
                dest = this2
                v = (x / 2)
                c1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = 0.
                c2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = 0.
                c3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
                v = (x / 2)
                s1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                v = 0.
                s2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                v = 0.
                s3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
                dest.x = (((s1 * c2) * c3) + (((c1 * s2) * s3)))
                dest.y = (((c1 * s2) * c3) - (((s1 * c2) * s3)))
                dest.z = (((c1 * c2) * s3) + (((s1 * s2) * c3)))
                dest.w = (((c1 * c2) * c3) - (((s1 * s2) * s3)))
                x = 1
                y = 1
                z = 1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this2 = glm_Vec3Base()
                this2.x = x
                this2.y = y
                this2.z = z
                b1.setTransform(this1,dest,this2)
                b1 = b
                x = -1
                y = -1
                z = -1
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = b1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(-0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 99, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo(-0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 100, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(-0.5,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 101, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_10)
            _gthis.it("should calculate support vertices when rotated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 95, 'className': "TestShapes3D", 'methodName': "new"}))
        self.describe("Using A Box",buddy_TestFunc.Sync(_hx_local_11))
        def _hx_local_16():
            s = None
            def _hx_local_12():
                nonlocal s
                x = 0
                y = 0
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                s = headbutt_threed_shapes_Sphere(this1,1)
            tmp = buddy_TestFunc.Sync(_hx_local_12)
            _gthis.beforeEach(tmp)
            def _hx_local_13():
                x = 4
                y = 1
                z = 3
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                s.position = this1
                c = s.get_centre()
                buddy_ShouldFloat.should(c.x).beCloseTo(4,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 114, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.y).beCloseTo(1,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 115, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(c.z).beCloseTo(3,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 116, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_13)
            _gthis.it("should calculate moved centres when transformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 111, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_14():
                s1 = s
                x = 1
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = s1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 121, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 122, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 123, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_14)
            _gthis.it("should calculate support vertices when untransformed",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 119, 'className': "TestShapes3D", 'methodName': "new"}))
            def _hx_local_15():
                s.position.x = 3
                s1 = s
                x = 1
                y = 1
                z = 0
                if (z is None):
                    z = 0
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec3Base()
                this1.x = x
                this1.y = y
                this1.z = z
                support = s1.support(this1)
                buddy_ShouldFloat.should(support.x).beCloseTo(((1 / python_lib_Math.sqrt(2)) + 3),None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 130, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.y).beCloseTo((1 / python_lib_Math.sqrt(2)),None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 131, 'className': "TestShapes3D", 'methodName': "new"}))
                buddy_ShouldFloat.should(support.z).beCloseTo(0,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 132, 'className': "TestShapes3D", 'methodName': "new"}))
            tmp = buddy_TestFunc.Sync(_hx_local_15)
            _gthis.it("should calculate support vertices when translated",tmp,None,_hx_AnonObject({'fileName': "test/TestShapes3D.hx", 'lineNumber': 126, 'className': "TestShapes3D", 'methodName': "new"}))
        self.describe("Using A Sphere",buddy_TestFunc.Sync(_hx_local_16))
TestShapes3D._hx_class = TestShapes3D


class Type:
    _hx_class_name = "Type"
    __slots__ = ()
    _hx_statics = ["getClass", "getClassName", "enumEq"]

    @staticmethod
    def getClass(o):
        if (o is None):
            return None
        o1 = o
        if ((o1 is not None) and ((HxOverrides.eq(o1,str) or python_lib_Inspect.isclass(o1)))):
            return None
        if isinstance(o,_hx_AnonObject):
            return None
        if hasattr(o,"_hx_class"):
            return o._hx_class
        if hasattr(o,"__class__"):
            return o.__class__
        else:
            return None

    @staticmethod
    def getClassName(c):
        if hasattr(c,"_hx_class_name"):
            return c._hx_class_name
        else:
            if (c == list):
                return "Array"
            if (c == Math):
                return "Math"
            if (c == str):
                return "String"
            try:
                return c.__name__
            except BaseException as _g:
                None
                return None

    @staticmethod
    def enumEq(a,b):
        if HxOverrides.eq(a,b):
            return True
        try:
            if ((b is None) and (not HxOverrides.eq(a,b))):
                return False
            if (a.tag != b.tag):
                return False
            p1 = a.params
            p2 = b.params
            if (len(p1) != len(p2)):
                return False
            _g = 0
            _g1 = len(p1)
            while (_g < _g1):
                i = _g
                _g = (_g + 1)
                if (not Type.enumEq(p1[i],p2[i])):
                    return False
            if (a._hx_class != b._hx_class):
                return False
        except BaseException as _g:
            None
            return False
        return True
Type._hx_class = Type

class buddy_SpecStatus(Enum):
    __slots__ = ()
    _hx_class_name = "buddy.SpecStatus"
    _hx_constructs = ["Unknown", "Passed", "Pending", "Failed"]
buddy_SpecStatus.Unknown = buddy_SpecStatus("Unknown", 0, ())
buddy_SpecStatus.Passed = buddy_SpecStatus("Passed", 1, ())
buddy_SpecStatus.Pending = buddy_SpecStatus("Pending", 2, ())
buddy_SpecStatus.Failed = buddy_SpecStatus("Failed", 3, ())
buddy_SpecStatus._hx_class = buddy_SpecStatus

class buddy_Step(Enum):
    __slots__ = ()
    _hx_class_name = "buddy.Step"
    _hx_constructs = ["TSuite", "TSpec"]

    @staticmethod
    def TSuite(s):
        return buddy_Step("TSuite", 0, (s,))

    @staticmethod
    def TSpec(s):
        return buddy_Step("TSpec", 1, (s,))
buddy_Step._hx_class = buddy_Step


class buddy_Suite:
    _hx_class_name = "buddy.Suite"
    __slots__ = ("description", "steps", "error", "stack")
    _hx_fields = ["description", "steps", "error", "stack"]
    _hx_methods = ["get_specs", "get_suites", "get_time", "passed"]

    def __init__(self,description):
        self.error = None
        self.stack = list()
        self.steps = list()
        if (description is None):
            raise haxe_Exception.thrown("Suite requires a description.")
        self.description = description

    def get_specs(self):
        output = []
        _g = 0
        _g1 = self.steps
        while (_g < len(_g1)):
            step = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            if (step.index == 1):
                s = step.params[0]
                output.append(s)
        return output

    def get_suites(self):
        output = []
        _g = 0
        _g1 = self.steps
        while (_g < len(_g1)):
            step = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            if (step.index == 0):
                s = step.params[0]
                output.append(s)
        return output

    def get_time(self):
        total = 0.0
        _g = 0
        _g1 = self.steps
        while (_g < len(_g1)):
            step = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            tmp = step.index
            if (tmp == 0):
                s = step.params[0]
                total = (total + s.get_time())
            elif (tmp == 1):
                s1 = step.params[0]
                total = (total + s1.time)
            else:
                pass
        return total

    def passed(self):
        def _hx_local_0(spec):
            return (spec.status == buddy_SpecStatus.Failed)
        if Lambda.exists(self.get_specs(),_hx_local_0):
            return False
        def _hx_local_2():
            def _hx_local_1(suite):
                return (not suite.passed())
            return (not Lambda.exists(self.get_suites(),_hx_local_1))
        return _hx_local_2()

buddy_Suite._hx_class = buddy_Suite


class buddy_Spec:
    _hx_class_name = "buddy.Spec"
    __slots__ = ("description", "status", "failures", "traces", "fileName", "time")
    _hx_fields = ["description", "status", "failures", "traces", "fileName", "time"]

    def __init__(self,description,fileName):
        self.time = 0
        self.fileName = ""
        self.traces = list()
        self.failures = list()
        self.status = buddy_SpecStatus.Unknown
        if (description is None):
            raise haxe_Exception.thrown("Spec must have a description.")
        self.description = description
        self.fileName = fileName

buddy_Spec._hx_class = buddy_Spec


class buddy_Failure:
    _hx_class_name = "buddy.Failure"
    __slots__ = ("error", "stack")
    _hx_fields = ["error", "stack"]

    def __init__(self,error,stack):
        if (error is None):
            raise haxe_Exception.thrown("Failure must have an error.")
        self.error = error
        self.stack = ([] if ((stack is None)) else stack)

buddy_Failure._hx_class = buddy_Failure

class buddy_TestFunc(Enum):
    __slots__ = ()
    _hx_class_name = "buddy.TestFunc"
    _hx_constructs = ["Async", "Sync"]

    @staticmethod
    def Async(f):
        return buddy_TestFunc("Async", 0, (f,))

    @staticmethod
    def Sync(f):
        return buddy_TestFunc("Sync", 1, (f,))
buddy_TestFunc._hx_class = buddy_TestFunc

class buddy_TestSpec(Enum):
    __slots__ = ()
    _hx_class_name = "buddy.TestSpec"
    _hx_constructs = ["Describe", "It"]

    @staticmethod
    def Describe(suite,included):
        return buddy_TestSpec("Describe", 0, (suite,included))

    @staticmethod
    def It(description,test,included,pos,time):
        return buddy_TestSpec("It", 1, (description,test,included,pos,time))
buddy_TestSpec._hx_class = buddy_TestSpec


class buddy_TestSuite:
    _hx_class_name = "buddy.TestSuite"
    __slots__ = ("description", "beforeAll", "beforeEach", "specs", "afterEach", "afterAll")
    _hx_fields = ["description", "beforeAll", "beforeEach", "specs", "afterEach", "afterAll"]

    def __init__(self,description):
        self.afterAll = haxe_ds_List()
        self.afterEach = haxe_ds_List()
        self.specs = haxe_ds_List()
        self.beforeEach = haxe_ds_List()
        self.beforeAll = haxe_ds_List()
        if (description is None):
            raise haxe_Exception.thrown("TestSuite must have a description. Can be empty.")
        self.description = description

buddy_TestSuite._hx_class = buddy_TestSuite


class buddy_Should:
    _hx_class_name = "buddy.Should"
    __slots__ = ("value", "inverse")
    _hx_fields = ["value", "inverse"]
    _hx_methods = ["be", "beType", "quote", "fail", "test"]

    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        self.value = value
        self.inverse = inverse

    def be(self,expected,p = None):
        result = ((python_lib_Builtins.id(self.value) == python_lib_Builtins.id(expected)) if ((isinstance(self.value,list) and isinstance(expected,list))) else HxOverrides.eq(self.value,expected))
        self.test(result,p,((("Expected " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),(("Didn't expect " + HxOverrides.stringOrNull(self.quote(expected))) + " but was equal to that"))

    def beType(self,_hx_type,p = None):
        self.test(Std.isOfType(self.value,_hx_type),p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to be type ") + HxOverrides.stringOrNull(self.quote(_hx_type))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to be type ") + HxOverrides.stringOrNull(self.quote(_hx_type))))

    def quote(self,v):
        if Std.isOfType(v,str):
            return (("\"" + Std.string(v)) + "\"")
        if Std.isOfType(v,haxe_ds_List):
            return Std.string(Lambda.array(v))
        return Std.string(v)

    def fail(self,error,errorInverted,p):
        tmp = (errorInverted if (self.inverse) else error)
        tmp1 = buddy_SuitesRunner.posInfosToStack(p)
        buddy_SuitesRunner.currentTest(False,tmp,tmp1)

    def test(self,expr,p,error,errorInverted):
        if (buddy_SuitesRunner.currentTest is None):
            raise haxe_Exception.thrown("SuitesRunner.currentTest was null")
        if (not self.inverse):
            tmp = buddy_SuitesRunner.posInfosToStack(p)
            buddy_SuitesRunner.currentTest(expr,error,tmp)
        else:
            tmp = buddy_SuitesRunner.posInfosToStack(p)
            buddy_SuitesRunner.currentTest((not expr),errorInverted,tmp)

buddy_Should._hx_class = buddy_Should


class buddy_ShouldDynamic(buddy_Should):
    _hx_class_name = "buddy.ShouldDynamic"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldDynamic(self.value,(not self.inverse))

    @staticmethod
    def should(d):
        return buddy_ShouldDynamic(d)

buddy_ShouldDynamic._hx_class = buddy_ShouldDynamic


class buddy_ShouldEnum(buddy_Should):
    _hx_class_name = "buddy.ShouldEnum"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "be", "equal"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldEnum(self.value,(not self.inverse))

    def be(self,expected,p = None):
        self.equal(expected,p)

    def equal(self,expected,p = None):
        self.test(Type.enumEq(self.value,expected),p,((("Expected " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),(("Didn't expect " + HxOverrides.stringOrNull(self.quote(self.value))) + " but was equal to that"))

    @staticmethod
    def should(e):
        return buddy_ShouldEnum(e)

buddy_ShouldEnum._hx_class = buddy_ShouldEnum


class buddy_ShouldInt(buddy_Should):
    _hx_class_name = "buddy.ShouldInt"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "beLessThan", "beLessThanOrEqualTo", "beGreaterThan", "beGreaterThanOrEqualTo"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldInt(self.value,(not self.inverse))

    def beLessThan(self,expected,p = None):
        self.test((self.value < expected),p,((("Expected less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beLessThanOrEqualTo(self,expected,p = None):
        self.test((self.value <= expected),p,((("Expected less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThan(self,expected,p = None):
        self.test((self.value > expected),p,((("Expected greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThanOrEqualTo(self,expected,p = None):
        self.test((self.value >= expected),p,((("Expected greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    @staticmethod
    def should(i):
        return buddy_ShouldInt(i)

buddy_ShouldInt._hx_class = buddy_ShouldInt


class buddy_ShouldInt64(buddy_Should):
    _hx_class_name = "buddy.ShouldInt64"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "be", "beLessThan", "beLessThanOrEqualTo", "beGreaterThan", "beGreaterThanOrEqualTo"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldInt64(self.value,(not self.inverse))

    def be(self,expected,p = None):
        b = self.value
        v = (((expected.high - b.high) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
        if (v == 0):
            v = haxe__Int32_Int32_Impl_.ucompare(expected.low,b.low)
        result = ((((v if ((b.high < 0)) else -1) if ((expected.high < 0)) else (v if ((b.high >= 0)) else 1))) == 0)
        self.test(result,p,((("Expected " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),(("Didn't expect " + HxOverrides.stringOrNull(self.quote(expected))) + " but was equal to that"))

    def beLessThan(self,expected,p = None):
        a = self.value
        v = (((a.high - expected.high) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
        if (v == 0):
            v = haxe__Int32_Int32_Impl_.ucompare(a.low,expected.low)
        self.test(((((v if ((expected.high < 0)) else -1) if ((a.high < 0)) else (v if ((expected.high >= 0)) else 1))) < 0),p,((("Expected less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beLessThanOrEqualTo(self,expected,p = None):
        a = self.value
        v = (((a.high - expected.high) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
        if (v == 0):
            v = haxe__Int32_Int32_Impl_.ucompare(a.low,expected.low)
        self.test(((((v if ((expected.high < 0)) else -1) if ((a.high < 0)) else (v if ((expected.high >= 0)) else 1))) <= 0),p,((("Expected less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThan(self,expected,p = None):
        a = self.value
        v = (((a.high - expected.high) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
        if (v == 0):
            v = haxe__Int32_Int32_Impl_.ucompare(a.low,expected.low)
        self.test(((((v if ((expected.high < 0)) else -1) if ((a.high < 0)) else (v if ((expected.high >= 0)) else 1))) > 0),p,((("Expected greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThanOrEqualTo(self,expected,p = None):
        a = self.value
        v = (((a.high - expected.high) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
        if (v == 0):
            v = haxe__Int32_Int32_Impl_.ucompare(a.low,expected.low)
        self.test(((((v if ((expected.high < 0)) else -1) if ((a.high < 0)) else (v if ((expected.high >= 0)) else 1))) >= 0),p,((("Expected greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    @staticmethod
    def should(i):
        return buddy_ShouldInt64(i)

buddy_ShouldInt64._hx_class = buddy_ShouldInt64


class buddy_ShouldFloat(buddy_Should):
    _hx_class_name = "buddy.ShouldFloat"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "beLessThan", "beLessThanOrEqualTo", "beGreaterThan", "beGreaterThanOrEqualTo", "beCloseTo"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldFloat(self.value,(not self.inverse))

    def beLessThan(self,expected,p = None):
        self.test((self.value < expected),p,((("Expected less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beLessThanOrEqualTo(self,expected,p = None):
        self.test((self.value <= expected),p,((("Expected less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not less than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThan(self,expected,p = None):
        self.test((self.value > expected),p,((("Expected greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beGreaterThanOrEqualTo(self,expected,p = None):
        self.test((self.value >= expected),p,((("Expected greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected not greater than or equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beCloseTo(self,expected,precision = None,p = None):
        if (precision is None):
            precision = 2
        diff = Reflect.field(Math,"fabs")((expected - self.value))
        threshold = (Math.pow(10,-precision) / 2)
        expr = (diff < threshold)
        self.test(expr,p,((("Expected close to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to be close to ") + HxOverrides.stringOrNull(self.quote(expected))))

    @staticmethod
    def should(i):
        return buddy_ShouldFloat(i)

buddy_ShouldFloat._hx_class = buddy_ShouldFloat


class buddy_ShouldDate(buddy_Should):
    _hx_class_name = "buddy.ShouldDate"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "beOn", "beBefore", "beAfter", "beOnStr", "beBeforeStr", "beAfterStr"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldDate(self.value,(not self.inverse))

    def beOn(self,expected,p = None):
        self.test(((self.value.date.timestamp() * 1000) == ((expected.date.timestamp() * 1000))),p,((("Expected date equal to " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),("Expected date not equal to " + HxOverrides.stringOrNull(self.quote(expected))))

    def beBefore(self,expected,p = None):
        self.test(((self.value.date.timestamp() * 1000) < ((expected.date.timestamp() * 1000))),p,((("Expected date before " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected date not before " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beAfter(self,expected,p = None):
        self.test(((self.value.date.timestamp() * 1000) > ((expected.date.timestamp() * 1000))),p,((("Expected date after " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),((("Expected date not after " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))))

    def beOnStr(self,expected,p = None):
        self.beOn(Date.fromString(expected),p)

    def beBeforeStr(self,expected,p = None):
        self.beBefore(Date.fromString(expected),p)

    def beAfterStr(self,expected,p = None):
        self.beAfter(Date.fromString(expected),p)

    @staticmethod
    def should(i):
        return buddy_ShouldDate(i)

buddy_ShouldDate._hx_class = buddy_ShouldDate


class buddy_ShouldIterable(buddy_Should):
    _hx_class_name = "buddy.ShouldIterable"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "contain", "containAll", "containExactly"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldIterable(self.value,(not self.inverse))

    def contain(self,o,p = None):
        def _hx_local_0(el):
            return HxOverrides.eq(el,o)
        self.test(Lambda.exists(self.value,_hx_local_0),p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to contain ") + HxOverrides.stringOrNull(self.quote(o))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to contain ") + HxOverrides.stringOrNull(self.quote(o))))

    def containAll(self,values,p = None):
        expr = True
        a = HxOverrides.iterator(values)
        while a.hasNext():
            a1 = [a.next()]
            def _hx_local_1(a):
                def _hx_local_0(v):
                    return HxOverrides.eq(v,(a[0] if 0 < len(a) else None))
                return _hx_local_0
            if (not Lambda.exists(self.value,_hx_local_1(a1))):
                expr = False
                break
        self.test(expr,p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to contain all of ") + HxOverrides.stringOrNull(self.quote(values))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to contain all of ") + HxOverrides.stringOrNull(self.quote(values))))

    def containExactly(self,values,p = None):
        a = HxOverrides.iterator(self.value)
        b = HxOverrides.iterator(values)
        expr = True
        while (a.hasNext() or b.hasNext()):
            if not HxOverrides.eq(a.next(),b.next()):
                expr = False
                break
        self.test(expr,p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to contain exactly ") + HxOverrides.stringOrNull(self.quote(values))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to contain exactly ") + HxOverrides.stringOrNull(self.quote(values))))

    @staticmethod
    def should(value):
        return buddy_ShouldIterable(value)

buddy_ShouldIterable._hx_class = buddy_ShouldIterable


class buddy_ShouldString(buddy_Should):
    _hx_class_name = "buddy.ShouldString"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["get_not", "contain", "startWith", "endWith", "match"]
    _hx_statics = ["should"]
    _hx_interfaces = []
    _hx_super = buddy_Should


    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        super().__init__(value,inverse)

    def get_not(self):
        return buddy_ShouldString(self.value,(not self.inverse))

    def contain(self,substring,p = None):
        if (self.value is None):
            self.fail((("Expected string to contain " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),(("Expected string not to contain " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),p)
            return
        _this = self.value
        startIndex = None
        self.test((((_this.find(substring) if ((startIndex is None)) else HxString.indexOfImpl(_this,substring,startIndex))) >= 0),p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to contain ") + HxOverrides.stringOrNull(self.quote(substring))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to contain ") + HxOverrides.stringOrNull(self.quote(substring))))

    def startWith(self,substring,p = None):
        if (self.value is None):
            self.fail((("Expected string to start with " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),(("Expected string not to start with " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),p)
            return
        self.test(self.value.startswith(substring),p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to start with ") + HxOverrides.stringOrNull(self.quote(substring))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to start with ") + HxOverrides.stringOrNull(self.quote(substring))))

    def endWith(self,substring,p = None):
        if (self.value is None):
            self.fail((("Expected string to end with " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),(("Expected string not to end with " + HxOverrides.stringOrNull(self.quote(substring))) + " but string was null"),p)
            return
        self.test(self.value.endswith(substring),p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to end with ") + HxOverrides.stringOrNull(self.quote(substring))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to end with ") + HxOverrides.stringOrNull(self.quote(substring))))

    def match(self,regexp,p = None):
        if (self.value is None):
            self.fail("Expected string to match regular expression but string was null","Expected string not to match regular expression but string was null",p)
            return
        regexp.matchObj = python_lib_Re.search(regexp.pattern,self.value)
        self.test((regexp.matchObj is not None),p,(("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to match regular expression"),(("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to match regular expression"))

    @staticmethod
    def should(_hx_str):
        return buddy_ShouldString(_hx_str)

buddy_ShouldString._hx_class = buddy_ShouldString


class buddy_ShouldFunctions:
    _hx_class_name = "buddy.ShouldFunctions"
    __slots__ = ("value", "inverse")
    _hx_fields = ["value", "inverse"]
    _hx_methods = ["get_not", "throwAnything", "throwValue", "throwType", "be", "quote", "test"]
    _hx_statics = ["should"]

    def __init__(self,value,inverse = None):
        if (inverse is None):
            inverse = False
        self.value = value
        self.inverse = inverse

    def get_not(self):
        return buddy_ShouldFunctions(self.value,(not self.inverse))

    def throwAnything(self,p = None):
        caught = False
        exception = None
        try:
            self.value()
        except BaseException as _g:
            None
            e = haxe_Exception.caught(_g).unwrap()
            exception = e
            caught = True
        self.test(caught,p,(("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to throw anything, nothing was thrown"),(((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to throw anything, ") + HxOverrides.stringOrNull(self.quote(exception))) + " was thrown"))
        return exception

    def throwValue(self,v,p = None):
        exception = None
        try:
            self.value()
        except BaseException as _g:
            None
            e = haxe_Exception.caught(_g).unwrap()
            cause = None
            exception = (e if ((cause is None)) else cause)
        isCaught = HxOverrides.eq(exception,v)
        self.test(isCaught,p,((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to throw ") + HxOverrides.stringOrNull(self.quote(v))),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to throw ") + HxOverrides.stringOrNull(self.quote(v))))
        return exception

    def throwType(self,_hx_type,p = None):
        exception = None
        try:
            self.value()
        except BaseException as _g:
            None
            e = haxe_Exception.caught(_g).unwrap()
            cause = None
            exception = (e if ((cause is None)) else cause)
        typeName = Type.getClassName(_hx_type)
        exceptionName = (None if ((exception is None)) else Type.getClassName(Type.getClass(exception)))
        if (exceptionName is None):
            exceptionName = "no exception"
        isCaught = Std.isOfType(exception,_hx_type)
        self.test(isCaught,p,(((((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " to throw type ") + ("null" if typeName is None else typeName)) + ", ") + ("null" if exceptionName is None else exceptionName)) + " was thrown instead"),((("Expected " + HxOverrides.stringOrNull(self.quote(self.value))) + " not to throw type ") + ("null" if typeName is None else typeName)))
        return exception

    def be(self,expected,p = None):
        self.test((self.value == expected),p,((("Expected " + HxOverrides.stringOrNull(self.quote(expected))) + ", was ") + HxOverrides.stringOrNull(self.quote(self.value))),(("Didn't expect " + HxOverrides.stringOrNull(self.quote(expected))) + " but was equal to that"))

    def quote(self,v):
        if Std.isOfType(v,str):
            return (("\"" + Std.string(v)) + "\"")
        if Std.isOfType(v,haxe_ds_List):
            return Std.string(Lambda.array(v))
        return Std.string(v)

    def test(self,expr,p,error,errorInverted):
        if (buddy_SuitesRunner.currentTest is None):
            raise haxe_Exception.thrown("SuitesRunner.currentTest was null")
        if (not self.inverse):
            tmp = buddy_SuitesRunner.posInfosToStack(p)
            buddy_SuitesRunner.currentTest(expr,error,tmp)
        else:
            tmp = buddy_SuitesRunner.posInfosToStack(p)
            buddy_SuitesRunner.currentTest((not expr),errorInverted,tmp)

    @staticmethod
    def should(value):
        return buddy_ShouldFunctions(value)

buddy_ShouldFunctions._hx_class = buddy_ShouldFunctions


class buddy_SuitesRunner:
    _hx_class_name = "buddy.SuitesRunner"
    __slots__ = ("unrecoverableError", "unrecoverableErrorStack", "allTestsPassed", "buddySuites", "reporter", "runCompleted", "includeMode", "oldLog")
    _hx_fields = ["unrecoverableError", "unrecoverableErrorStack", "allTestsPassed", "buddySuites", "reporter", "runCompleted", "includeMode", "oldLog"]
    _hx_methods = ["run", "runDescribes", "failed", "statusCode", "startRun", "startIncludeMode", "mapTestSuite", "runTestFuncs", "flatten", "isSync", "mapTestSpec", "haveUnrecoverableError"]
    _hx_statics = ["currentTest", "posInfosToStack"]

    def __init__(self,buddySuites,reporter = None):
        self.includeMode = None
        self.runCompleted = None
        self.allTestsPassed = False
        self.unrecoverableErrorStack = None
        self.unrecoverableError = None
        self.buddySuites = buddySuites
        self.reporter = (buddy_reporting_ConsoleReporter() if ((reporter is None)) else reporter)
        self.oldLog = haxe_Log.trace
        def _hx_local_0(suite):
            metaData = haxe_rtti_Meta.getType(Type.getClass(suite))
            return python_Boot.hasField(metaData,"includeMode")
        self.includeMode = Lambda.exists(buddySuites,_hx_local_0)

    def run(self):
        _gthis = self
        buddy_PythonSys.setrecursionlimit(100000)
        self.runCompleted = promhx_Deferred()
        runCompletedPromise = self.runCompleted.promise()
        def _hx_local_0(err):
            if (err is not None):
                _gthis.haveUnrecoverableError(err)
                return
            if _gthis.includeMode:
                _gthis.startIncludeMode()
            _gthis.startRun()
        self.runDescribes(_hx_local_0)
        return runCompletedPromise

    def runDescribes(self,cb):
        _gthis = self
        asyncQueue = list()
        syncQueue = list()
        def _hx_local_0(suite):
            while (not Lambda.empty(suite.describeQueue)):
                _this = suite.describeQueue
                current = (None if ((len(_this) == 0)) else _this.pop())
                _g = current.spec
                processSuiteDescribes = _g.index
                if (processSuiteDescribes == 0):
                    f = _g.params[0]
                    x = _hx_AnonObject({'buddySuite': suite, 'testSuite': current.suite, 'run': f})
                    asyncQueue.append(x)
                elif (processSuiteDescribes == 1):
                    f1 = _g.params[0]
                    x1 = _hx_AnonObject({'buddySuite': suite, 'testSuite': current.suite, 'run': f1})
                    syncQueue.append(x1)
                else:
                    pass
        processSuiteDescribes = _hx_local_0
        processBuddySuites = None
        def _hx_local_5():
            nonlocal syncQueue
            buddySuite = HxOverrides.iterator(_gthis.buddySuites)
            while buddySuite.hasNext():
                buddySuite1 = buddySuite.next()
                processSuiteDescribes(buddySuite1)
            if (len(syncQueue) > 0):
                try:
                    _g = 0
                    while (_g < len(syncQueue)):
                        test = (syncQueue[_g] if _g >= 0 and _g < len(syncQueue) else None)
                        _g = (_g + 1)
                        test.buddySuite.currentSuite = test.testSuite
                        test.run()
                except BaseException as _g:
                    None
                    err = haxe_Exception.caught(_g).unwrap()
                    cb(err)
                    return
                syncQueue = []
                processBuddySuites()
            elif (len(asyncQueue) > 0):
                def _hx_local_3(test,cb):
                    test.buddySuite.currentSuite = test.testSuite
                    def _hx_local_2():
                        cb(None)
                    test.run(_hx_local_2)
                def _hx_local_4(err):
                    nonlocal asyncQueue
                    if (err is not None):
                        cb(err)
                        return
                    asyncQueue = []
                    processBuddySuites()
                AsyncTools.aEachSeries(asyncQueue,_hx_local_3,_hx_local_4)
            else:
                cb(None)
        processBuddySuites = _hx_local_5
        processBuddySuites()

    def failed(self):
        return (not self.allTestsPassed)

    def statusCode(self):
        if self.failed():
            return 1
        else:
            return 0

    def startRun(self):
        _gthis = self
        r = self.reporter.start()
        def _hx_local_6(go):
            if (not go):
                r = _gthis.reporter.done([],False)
                def _hx_local_0(_):
                    _gthis.runCompleted.resolve(_gthis)
                r.then(_hx_local_0)
                return
            beforeEachStack = [[]]
            afterEachStack = [[]]
            def _hx_local_2(buddySuite,done):
                def _hx_local_1(err,suite):
                    if ((err is None) and ((suite is None))):
                        return
                    if (err is not None):
                        suite.error = err
                        suite.stack = haxe__CallStack_CallStack_Impl_.exceptionStack()
                    done(err,suite)
                suiteDone = _hx_local_1
                syncSuite = _gthis.mapTestSuite(buddySuite,buddySuite.suite,beforeEachStack,afterEachStack,suiteDone)
                if (syncSuite is not None):
                    suiteDone(syncSuite.error,syncSuite.suite)
            def _hx_local_5(err,suites):
                if (err is not None):
                    _gthis.haveUnrecoverableError(err)
                else:
                    def _hx_local_3(suite):
                        return (not suite.passed())
                    _gthis.allTestsPassed = (not Lambda.exists(suites,_hx_local_3))
                    r = _gthis.reporter.done(suites,_gthis.allTestsPassed)
                    def _hx_local_4(_):
                        _gthis.runCompleted.resolve(_gthis)
                    r.then(_hx_local_4)
            AsyncTools.aMapSeries(_gthis.buddySuites,_hx_local_2,_hx_local_5)
        r.then(_hx_local_6)

    def startIncludeMode(self):
        traverse = None
        def _hx_local_1(suite):
            def _hx_local_0(spec):
                traverse1 = spec.index
                if (traverse1 == 0):
                    suite = spec.params[0]
                    included = spec.params[1]
                    if included:
                        return True
                    else:
                        return traverse(suite)
                elif (traverse1 == 1):
                    _g = spec.params[1]
                    _g = spec.params[3]
                    _g = spec.params[4]
                    desc = spec.params[0]
                    included = spec.params[2]
                    return included
                else:
                    pass
            suite.specs = suite.specs.filter(_hx_local_0)
            return (suite.specs.length > 0)
        traverse = _hx_local_1
        def _hx_local_2(buddySuite):
            suiteMeta = haxe_rtti_Meta.getType(Type.getClass(buddySuite))
            if python_Boot.hasField(suiteMeta,"include"):
                return True
            return traverse(buddySuite.suite)
        self.buddySuites = Lambda.filter(self.buddySuites,_hx_local_2)

    def mapTestSuite(self,buddySuite,testSuite,beforeEachStack,afterEachStack,done):
        _gthis = self
        def _hx_local_0():
            buddy_tests_SelfTest.lastSuite = buddy_Suite(testSuite.description)
            return buddy_tests_SelfTest.lastSuite
        currentSuite = _hx_local_0()
        x = Lambda.array(testSuite.beforeEach)
        beforeEachStack.append(x)
        x = Lambda.array(testSuite.afterEach)
        afterEachStack.insert(0, x)
        allSync = (self.isSync(testSuite.beforeAll) and self.isSync(testSuite.afterAll))
        result = None
        syncResultCount = 0
        def _hx_local_5(err):
            nonlocal result
            if (err is not None):
                if _gthis.isSync(testSuite.beforeAll):
                    result = _hx_AnonObject({'error': err, 'suite': currentSuite})
                else:
                    done(err,currentSuite)
                return
            def _hx_local_2(testSpec,cb):
                nonlocal syncResultCount
                result2 = _gthis.mapTestSpec(buddySuite,testSuite,beforeEachStack,afterEachStack,testSpec,cb)
                if (result2 is not None):
                    syncResultCount = (syncResultCount + 1)
                    cb(result2.error,result2.step)
            def _hx_local_4(err,testSteps):
                nonlocal allSync
                nonlocal result
                allSync = (allSync and ((len(testSteps) == syncResultCount)))
                if (err is not None):
                    if allSync:
                        result = _hx_AnonObject({'error': err, 'suite': currentSuite})
                    else:
                        done(err,currentSuite)
                    return
                def _hx_local_3(err):
                    nonlocal result
                    nonlocal result
                    if (err is not None):
                        if allSync:
                            result = _hx_AnonObject({'error': err, 'suite': currentSuite})
                        else:
                            done(err,currentSuite)
                        return
                    currentSuite.steps = testSteps
                    if (len(beforeEachStack) != 0):
                        beforeEachStack.pop()
                    if (len(afterEachStack) != 0):
                        afterEachStack.pop(0)
                    if allSync:
                        result = _hx_AnonObject({'error': None, 'suite': currentSuite})
                    else:
                        done(None,currentSuite)
                _gthis.runTestFuncs(testSuite.afterAll,_hx_local_3)
            AsyncTools.aMapSeries(testSuite.specs,_hx_local_2,_hx_local_4)
        self.runTestFuncs(testSuite.beforeAll,_hx_local_5)
        if (result is not None):
            done(None,None)
        return result

    def runTestFuncs(self,funcs,done):
        syncQ = []
        asyncQ = []
        func = HxOverrides.iterator(funcs)
        while func.hasNext():
            func1 = func.next()
            tmp = func1.index
            if (tmp == 0):
                f = func1.params[0]
                asyncQ.append(f)
            elif (tmp == 1):
                f1 = func1.params[0]
                syncQ.append(f1)
            else:
                pass
        try:
            _g = 0
            while (_g < len(syncQ)):
                f = (syncQ[_g] if _g >= 0 and _g < len(syncQ) else None)
                _g = (_g + 1)
                f()
        except BaseException as _g:
            None
            err = haxe_Exception.caught(_g).unwrap()
            done(err)
            return
        def _hx_local_2(f,done):
            def _hx_local_1():
                done()
            f(_hx_local_1)
        AsyncTools.aEachSeries(asyncQ,_hx_local_2,done)

    def flatten(self,arr):
        _g = []
        _g1 = 0
        while (_g1 < len(arr)):
            a = (arr[_g1] if _g1 >= 0 and _g1 < len(arr) else None)
            _g1 = (_g1 + 1)
            _g2 = 0
            while (_g2 < len(a)):
                b = (a[_g2] if _g2 >= 0 and _g2 < len(a) else None)
                _g2 = (_g2 + 1)
                _g.append(b)
        return _g

    def isSync(self,funcs):
        f = HxOverrides.iterator(funcs)
        while f.hasNext():
            f1 = f.next()
            if (f1.index == 0):
                _g = f1.params[0]
                return False
        return True

    def mapTestSpec(self,buddySuite,testSuite,beforeEachStack,afterEachStack,testSpec,done):
        _gthis = self
        hasCompleted = False
        oldFail = None
        def _hx_local_1():
            def _hx_local_0(err = None,p = None):
                if (err is None):
                    err = "Exception"
                if ((not hasCompleted) and ((oldFail == buddySuite.fail))):
                    done(err,None)
            buddySuite.fail = _hx_local_0
            return buddySuite.fail
        oldFail = _hx_local_1()
        def _hx_local_3():
            def _hx_local_2(message = None,p = None):
                done("Cannot call pending here.",None)
            buddySuite.pending = _hx_local_2
            return buddySuite.pending
        oldPending = _hx_local_3()
        tmp = testSpec.index
        if (tmp == 0):
            _g = testSpec.params[1]
            testSuite = testSpec.params[0]
            def _hx_local_4(err,newSuite):
                if ((err is None) and ((newSuite is None))):
                    return
                if (err is not None):
                    done(err,None)
                else:
                    done(None,buddy_Step.TSuite(newSuite))
            result = self.mapTestSuite(buddySuite,testSuite,beforeEachStack,afterEachStack,_hx_local_4)
            if (result is not None):
                return _hx_AnonObject({'error': result.error, 'step': buddy_Step.TSuite(result.suite)})
            else:
                return None
        elif (tmp == 1):
            _g = testSpec.params[2]
            desc = testSpec.params[0]
            test = testSpec.params[1]
            pos = testSpec.params[3]
            time = testSpec.params[4]
            def _hx_local_5():
                buddy_tests_SelfTest.lastSpec = buddy_Spec(desc,pos.fileName)
                return buddy_tests_SelfTest.lastSpec
            spec = _hx_local_5()
            beforeEach = self.flatten(beforeEachStack)
            afterEach = self.flatten(afterEachStack)
            eachIsSync = (self.isSync(beforeEach) and self.isSync(afterEach))
            returnSync = None
            if (test is None):
                returnSync = eachIsSync
            else:
                returnSync1 = test.index
                if (returnSync1 == 0):
                    _g = test.params[0]
                    returnSync = False
                elif (returnSync1 == 1):
                    _g = test.params[0]
                    returnSync = eachIsSync
                else:
                    pass
            if (not buddy_BuddySuite.useDefaultTrace):
                def _hx_local_7(v,pos = None):
                    if (pos is None):
                        _this = spec.traces
                        x = Std.string(v)
                        _this.append(x)
                    else:
                        output = None
                        if (Reflect.field(pos,"customParams") is not None):
                            output1 = (Std.string(v) + ",")
                            def _hx_local_6(v2):
                                return Std.string(v2)
                            _this = list(map(_hx_local_6,Reflect.field(pos,"customParams")))
                            output = (("null" if output1 is None else output1) + HxOverrides.stringOrNull(",".join([python_Boot.toString1(x1,'') for x1 in _this])))
                        else:
                            output = Std.string(v)
                        _this = spec.traces
                        x = ((((HxOverrides.stringOrNull(pos.fileName) + ":") + Std.string(pos.lineNumber)) + ": ") + ("null" if output is None else output))
                        _this.append(x)
                haxe_Log.trace = _hx_local_7
            def _hx_local_8(error,stack):
                if hasCompleted:
                    return
                spec.status = buddy_SpecStatus.Failed
                _this = spec.failures
                x = buddy_Failure(error,stack)
                _this.append(x)
            reportFailure = _hx_local_8
            def _hx_local_11(status):
                nonlocal hasCompleted
                if hasCompleted:
                    return None
                hasCompleted = True
                if (spec.status == buddy_SpecStatus.Unknown):
                    spec.status = status
                if (not buddy_BuddySuite.useDefaultTrace):
                    haxe_Log.trace = _gthis.oldLog
                buddySuite.fail = oldFail
                buddySuite.pending = oldPending
                syncResult = None
                def _hx_local_10(err):
                    nonlocal syncResult
                    if returnSync:
                        syncResult = _hx_AnonObject({'error': err, 'step': (buddy_Step.TSpec(spec) if ((err is None)) else None)})
                        _gthis.reporter.progress(spec)
                    elif (err is not None):
                        done(err,None)
                    else:
                        r = _gthis.reporter.progress(spec)
                        def _hx_local_9(_):
                            done(None,buddy_Step.TSpec(spec))
                        r.then(_hx_local_9)
                _gthis.runTestFuncs(afterEach,_hx_local_10)
                return syncResult
            specCompleted = _hx_local_11
            if (test is None):
                return specCompleted(buddy_SpecStatus.Pending)
            def _hx_local_12(testStatus,error,stack):
                if (testStatus != True):
                    reportFailure(error,stack)
            buddy_SuitesRunner.currentTest = _hx_local_12
            if ((not returnSync) and ((buddySuite.timeoutMs > 0))):
                r = buddy_tools_AsyncTools.wait(buddySuite.timeoutMs)
                def _hx_local_13(e):
                    reportFailure(e,haxe__CallStack_CallStack_Impl_.exceptionStack())
                    specCompleted(buddy_SpecStatus.Failed)
                r.catchError(_hx_local_13)
                def _hx_local_14(_):
                    reportFailure((("Timeout after " + Std.string(buddySuite.timeoutMs)) + " ms"),[])
                    specCompleted(buddy_SpecStatus.Failed)
                r.then(_hx_local_14)
            _syncResult = None
            _startTime = python_lib_Timeit.default_timer()
            def _hx_local_15(status):
                nonlocal _syncResult
                if ((not returnSync) or ((_syncResult is not None))):
                    return
                _syncResult = status
                setSyncResult = python_lib_Timeit.default_timer()
                spec.time = (setSyncResult - _startTime)
            setSyncResult = _hx_local_15
            def _hx_local_16(err = None,p = None):
                if (err is None):
                    err = "Manually"
                reportFailure(err,buddy_SuitesRunner.posInfosToStack(p))
                setSyncResult(specCompleted(buddy_SpecStatus.Failed))
            buddySuite.fail = _hx_local_16
            def _hx_local_17(message = None,p = None):
                msg = (((HxOverrides.stringOrNull(p.fileName) + ":") + Std.string(p.lineNumber)) + HxOverrides.stringOrNull((((": " + ("null" if message is None else message)) if ((message is not None)) else ""))))
                _this = spec.traces
                _this.append(msg)
                setSyncResult(specCompleted(buddy_SpecStatus.Pending))
            buddySuite.pending = _hx_local_17
            def _hx_local_21(err):
                if (err is not None):
                    if returnSync:
                        setSyncResult(_hx_AnonObject({'error': err, 'step': None}))
                    else:
                        done(err,None)
                    return
                def _hx_local_19(func,done):
                    try:
                        runTestFunc = func.index
                        if (runTestFunc == 0):
                            f = func.params[0]
                            def _hx_local_18():
                                done(None)
                            f(_hx_local_18)
                        elif (runTestFunc == 1):
                            f = func.params[0]
                            f()
                            done(None)
                        else:
                            pass
                    except BaseException as _g:
                        None
                        e = haxe_Exception.caught(_g).unwrap()
                        done(e)
                runTestFunc = _hx_local_19
                def _hx_local_20(err):
                    if (err is not None):
                        reportFailure(err,haxe__CallStack_CallStack_Impl_.exceptionStack())
                        setSyncResult(specCompleted(buddy_SpecStatus.Failed))
                    else:
                        setSyncResult(specCompleted(buddy_SpecStatus.Passed))
                runTestFunc(test,_hx_local_20)
            self.runTestFuncs(beforeEach,_hx_local_21)
            return _syncResult
        else:
            pass

    def haveUnrecoverableError(self,err):
        self.unrecoverableError = err
        self.unrecoverableErrorStack = haxe__CallStack_CallStack_Impl_.exceptionStack()
        self.runCompleted.resolve(self)
    currentTest = None

    @staticmethod
    def posInfosToStack(p):
        if (p is None):
            return [haxe_StackItem.FilePos(None,"",0)]
        else:
            return [haxe_StackItem.FilePos(None,p.fileName,p.lineNumber)]

buddy_SuitesRunner._hx_class = buddy_SuitesRunner


class buddy_internal_GenerateMain:
    _hx_class_name = "buddy.internal.GenerateMain"
    __slots__ = ()
    _hx_statics = ["testsRunning"]
    testsRunning = None
buddy_internal_GenerateMain._hx_class = buddy_internal_GenerateMain


class buddy_reporting_Reporter:
    _hx_class_name = "buddy.reporting.Reporter"
    __slots__ = ()
    _hx_methods = ["start", "progress", "done"]
buddy_reporting_Reporter._hx_class = buddy_reporting_Reporter


class buddy_reporting_TraceReporter:
    _hx_class_name = "buddy.reporting.TraceReporter"
    __slots__ = ("colors",)
    _hx_fields = ["colors"]
    _hx_methods = ["start", "progress", "done", "print", "println", "strCol", "resolveImmediately"]
    _hx_interfaces = [buddy_reporting_Reporter]

    def __init__(self,colors = None):
        if (colors is None):
            colors = False
        self.colors = colors

    def start(self):
        return self.resolveImmediately(True)

    def progress(self,spec):
        return self.resolveImmediately(spec)

    def done(self,suites,status):
        _gthis = self
        self.println("")
        total = 0
        failures = 0
        pending = 0
        countTests = None
        printTests = None
        def _hx_local_5(s):
            nonlocal total
            nonlocal pending
            nonlocal failures
            nonlocal failures
            if (s.error is not None):
                failures = (failures + 1)
            _g = 0
            _g1 = s.steps
            while (_g < len(_g1)):
                sp = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                _g = (_g + 1)
                countTests1 = sp.index
                if (countTests1 == 0):
                    s = sp.params[0]
                    countTests(s)
                elif (countTests1 == 1):
                    sp1 = sp.params[0]
                    total = (total + 1)
                    if (sp1.status == buddy_SpecStatus.Failed):
                        failures = (failures + 1)
                    elif (sp1.status == buddy_SpecStatus.Pending):
                        pending = (pending + 1)
                else:
                    pass
        countTests = _hx_local_5
        Lambda.iter(suites,countTests)
        def _hx_local_13(s,indentLevel):
            success = True
            lines = []
            def _hx_local_6(_hx_str,color = None):
                if (color is None):
                    color = 39
                start = _gthis.strCol(color)
                end = _gthis.strCol(39)
                str1 = len(_hx_str)
                b = (indentLevel * 2)
                x = (0 if (python_lib_Math.isnan(0)) else (b if (python_lib_Math.isnan(b)) else max(0,b)))
                x1 = None
                try:
                    x1 = int(x)
                except BaseException as _g:
                    None
                    x1 = None
                x = ((("null" if start is None else start) + HxOverrides.stringOrNull(StringTools.lpad(_hx_str," ",(str1 + x1)))) + ("null" if end is None else end))
                lines.append(x)
            _hx_print = _hx_local_6
            def _hx_local_8(indent,stack):
                if ((stack is None) or ((len(stack) == 0))):
                    return
                _g = 0
                while (_g < len(stack)):
                    s = (stack[_g] if _g >= 0 and _g < len(stack) else None)
                    _g = (_g + 1)
                    if (s.index == 2):
                        _g1 = s.params[0]
                        _g2 = s.params[3]
                        file = s.params[1]
                        line = s.params[2]
                        printStack = None
                        printStack1 = None
                        if (line > 0):
                            startIndex = None
                            printStack1 = (((file.find("buddy/internal/") if ((startIndex is None)) else HxString.indexOfImpl(file,"buddy/internal/",startIndex))) != 0)
                        else:
                            printStack1 = False
                        if printStack1:
                            startIndex1 = None
                            printStack = (((file.find("buddy.SuitesRunner") if ((startIndex1 is None)) else HxString.indexOfImpl(file,"buddy.SuitesRunner",startIndex1))) != 0)
                        else:
                            printStack = False
                        if printStack:
                            _hx_print((("null" if indent is None else indent) + HxOverrides.stringOrNull((((("@ " + ("null" if file is None else file)) + ":") + Std.string(line))))),33)
            printStack = _hx_local_8
            def _hx_local_10(spec):
                _g = 0
                _g1 = spec.traces
                while (_g < len(_g1)):
                    t = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                    _g = (_g + 1)
                    _hx_print(("    " + ("null" if t is None else t)),33)
            printTraces = _hx_local_10
            if (len(s.description) > 0):
                _hx_print(s.description)
            if (s.error is not None):
                _hx_print(("ERROR: " + Std.string(s.error)),31)
                printStack("  ",s.stack)
                return _hx_AnonObject({'success': False, 'lines': lines})
            _g = 0
            _g1 = s.steps
            while (_g < len(_g1)):
                step = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                _g = (_g + 1)
                printTests1 = step.index
                if (printTests1 == 0):
                    s = step.params[0]
                    ret = printTests(s,(indentLevel + 1))
                    success = (success and ret.success)
                    lines = (lines + ret.lines)
                elif (printTests1 == 1):
                    sp = step.params[0]
                    success = (success and ((sp.status == buddy_SpecStatus.Passed)))
                    if (sp.status == buddy_SpecStatus.Failed):
                        _hx_print((("  " + HxOverrides.stringOrNull(sp.description)) + " (FAILED)"),31)
                        printTraces(sp)
                        _g2 = 0
                        _g3 = sp.failures
                        while (_g2 < len(_g3)):
                            failure = (_g3[_g2] if _g2 >= 0 and _g2 < len(_g3) else None)
                            _g2 = (_g2 + 1)
                            _hx_print(("    " + Std.string(failure.error)),33)
                            printStack("      ",failure.stack)
                    else:
                        _hx_print((((("  " + HxOverrides.stringOrNull(sp.description)) + " (") + Std.string(sp.status)) + ")"),(32 if ((sp.status == buddy_SpecStatus.Passed)) else 33))
                        printTraces(sp)
                else:
                    pass
            return _hx_AnonObject({'success': success, 'lines': lines})
        printTests = _hx_local_13
        def _hx_local_14(s):
            ret = printTests(s,-1)
            Lambda.iter(ret.lines,_gthis.println)
        Lambda.iter(suites,_hx_local_14)
        totalColor = (31 if ((failures > 0)) else 32)
        pendingColor = (33 if ((pending > 0)) else totalColor)
        self.println(((((HxOverrides.stringOrNull(self.strCol(totalColor)) + HxOverrides.stringOrNull(((((("" + Std.string(total)) + " specs, ") + Std.string(failures)) + " failures, ")))) + HxOverrides.stringOrNull(self.strCol(pendingColor))) + HxOverrides.stringOrNull(((("" + Std.string(pending)) + " pending")))) + HxOverrides.stringOrNull(self.strCol(39))))
        return self.resolveImmediately(suites)

    def print(self,s):
        pass

    def println(self,s):
        haxe_Log.trace(s,_hx_AnonObject({'fileName': "buddy/reporting/TraceReporter.hx", 'lineNumber': 159, 'className': "buddy.reporting.TraceReporter", 'methodName': "println"}))

    def strCol(self,color):
        if self.colors:
            return buddy_reporting__TraceReporter_Color_Impl_.ansiCode(color)
        else:
            return ""

    def resolveImmediately(self,o):
        _hx_def = promhx_Deferred()
        pr = _hx_def.promise()
        _hx_def.resolve(o)
        return pr

buddy_reporting_TraceReporter._hx_class = buddy_reporting_TraceReporter


class buddy_reporting_ConsoleReporter(buddy_reporting_TraceReporter):
    _hx_class_name = "buddy.reporting.ConsoleReporter"
    __slots__ = ("progressString",)
    _hx_fields = ["progressString"]
    _hx_methods = ["start", "progress", "done", "print", "println"]
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_reporting_TraceReporter


    def __init__(self,colors = None):
        if (colors is None):
            colors = False
        self.progressString = ""
        super().__init__(colors)

    def start(self):
        return self.resolveImmediately(True)

    def progress(self,spec):
        status = None
        status1 = spec.status.index
        if (status1 == 0):
            status = (HxOverrides.stringOrNull(self.strCol(33)) + "?")
        elif (status1 == 1):
            status = (HxOverrides.stringOrNull(self.strCol(32)) + ".")
        elif (status1 == 2):
            status = (HxOverrides.stringOrNull(self.strCol(33)) + "P")
        elif (status1 == 3):
            status = (HxOverrides.stringOrNull(self.strCol(31)) + "X")
        else:
            pass
        _hx_local_0 = self
        _hx_local_1 = _hx_local_0.progressString
        _hx_local_0.progressString = (("null" if _hx_local_1 is None else _hx_local_1) + ("null" if status is None else status))
        _hx_local_0.progressString
        self.print((("null" if status is None else status) + HxOverrides.stringOrNull(self.strCol(39))))
        return self.resolveImmediately(spec)

    def done(self,suites,status):
        output = super().done(suites,status)
        return output

    def print(self,s):
        python_Lib.printString(Std.string(s))

    def println(self,s):
        _hx_str = Std.string(s)
        python_Lib.printString((("" + ("null" if _hx_str is None else _hx_str)) + HxOverrides.stringOrNull(python_Lib.lineEnd)))

buddy_reporting_ConsoleReporter._hx_class = buddy_reporting_ConsoleReporter


class buddy_reporting_ConsoleFileReporter(buddy_reporting_ConsoleReporter):
    _hx_class_name = "buddy.reporting.ConsoleFileReporter"
    __slots__ = ("lastFileName",)
    _hx_fields = ["lastFileName"]
    _hx_methods = ["progress"]
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = buddy_reporting_ConsoleReporter


    def __init__(self,colors = None):
        if (colors is None):
            colors = False
        self.lastFileName = None
        super().__init__(colors)

    def progress(self,spec):
        if (self.lastFileName != spec.fileName):
            if (self.lastFileName is not None):
                _hx_local_0 = self
                _hx_local_1 = _hx_local_0.progressString
                _hx_local_0.progressString = (("null" if _hx_local_1 is None else _hx_local_1) + "\n")
                _hx_local_0.progressString
                self.println("")
            _hx_local_2 = self
            _hx_local_3 = _hx_local_2.progressString
            _hx_local_2.progressString = (("null" if _hx_local_3 is None else _hx_local_3) + HxOverrides.stringOrNull(((HxOverrides.stringOrNull(spec.fileName) + ": "))))
            _hx_local_2.progressString
            self.print((HxOverrides.stringOrNull(spec.fileName) + ": "))
            self.lastFileName = spec.fileName
        status = None
        status1 = spec.status.index
        if (status1 == 0):
            status = (HxOverrides.stringOrNull(self.strCol(33)) + "?")
        elif (status1 == 1):
            status = (HxOverrides.stringOrNull(self.strCol(32)) + ".")
        elif (status1 == 2):
            status = (HxOverrides.stringOrNull(self.strCol(33)) + "P")
        elif (status1 == 3):
            status = (HxOverrides.stringOrNull(self.strCol(31)) + "X")
        else:
            pass
        _hx_local_4 = self
        _hx_local_5 = _hx_local_4.progressString
        _hx_local_4.progressString = (("null" if _hx_local_5 is None else _hx_local_5) + ("null" if status is None else status))
        _hx_local_4.progressString
        self.print((("null" if status is None else status) + HxOverrides.stringOrNull(self.strCol(39))))
        return self.resolveImmediately(spec)

buddy_reporting_ConsoleFileReporter._hx_class = buddy_reporting_ConsoleFileReporter


class buddy_reporting__TraceReporter_Color_Impl_:
    _hx_class_name = "buddy.reporting._TraceReporter.Color_Impl_"
    __slots__ = ()
    _hx_statics = ["Default", "Red", "Yellow", "Green", "White", "ansiCode"]

    @staticmethod
    def ansiCode(this1):
        return ("\x1B" + HxOverrides.stringOrNull(((("[" + Std.string(this1)) + "m"))))
buddy_reporting__TraceReporter_Color_Impl_._hx_class = buddy_reporting__TraceReporter_Color_Impl_


class buddy_tests_SelfTest:
    _hx_class_name = "buddy.tests.SelfTest"
    __slots__ = ()
    _hx_statics = ["lastSpec", "lastSuite", "passLastSpecIf", "setLastSpec"]

    @staticmethod
    def passLastSpecIf(expr,failReason):
        if expr:
            buddy_tests_SelfTest.setLastSpec(buddy_SpecStatus.Passed)
        else:
            buddy_tests_SelfTest.setLastSpec(buddy_SpecStatus.Failed)
            _this = buddy_tests_SelfTest.lastSpec.failures
            x = buddy_Failure(failReason,[])
            _this.append(x)

    @staticmethod
    def setLastSpec(status):
        Reflect.setProperty(buddy_tests_SelfTest.lastSpec,"status",status)
buddy_tests_SelfTest._hx_class = buddy_tests_SelfTest


class buddy_tools_AsyncTools:
    _hx_class_name = "buddy.tools.AsyncTools"
    __slots__ = ()
    _hx_statics = ["iterateAsyncBool", "iterateAsync", "wait", "next"]

    @staticmethod
    def iterateAsyncBool(it,action):
        return buddy_tools_AsyncTools.iterateAsync(it,action,True)

    @staticmethod
    def iterateAsync(it,action,resolveWith):
        finished = promhx_Deferred()
        pr = finished.promise()
        buddy_tools_AsyncTools.next(HxOverrides.iterator(it),action,finished,resolveWith)
        return pr

    @staticmethod
    def wait(ms):
        _hx_def = promhx_Deferred()
        pr = _hx_def.promise()
        def _hx_local_0():
            if (not pr._fulfilled):
                _hx_def.resolve(True)
        done = _hx_local_0
        buddy_tools_Timer((ms / 1000),done).start()
        return pr

    @staticmethod
    def next(it,action,_hx_def,resolveWith):
        if (not it.hasNext()):
            _hx_def.resolve(resolveWith)
        else:
            n = it.next()
            r = action(n)
            def _hx_local_0(_):
                buddy_tools_AsyncTools.next(it,action,_hx_def,resolveWith)
            r.then(_hx_local_0)
buddy_tools_AsyncTools._hx_class = buddy_tools_AsyncTools


class glm_GLM:
    _hx_class_name = "glm.GLM"
    __slots__ = ()
    _hx_statics = ["EPSILON", "lerp", "translate", "rotate", "scale", "transform", "lookAt", "perspective", "orthographic", "frustum"]

    @staticmethod
    def lerp(a,b,t):
        return (a + ((t * ((b - a)))))

    @staticmethod
    def translate(translation,dest):
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        dest._30 = translation.x
        dest._31 = translation.y
        dest._32 = translation.z
        return dest

    @staticmethod
    def rotate(rotation,dest):
        x2 = (rotation.x + rotation.x)
        y2 = (rotation.y + rotation.y)
        z2 = (rotation.z + rotation.z)
        xx = (rotation.x * x2)
        xy = (rotation.x * y2)
        xz = (rotation.x * z2)
        yy = (rotation.y * y2)
        yz = (rotation.y * z2)
        zz = (rotation.z * z2)
        wx = (rotation.w * x2)
        wy = (rotation.w * y2)
        wz = (rotation.w * z2)
        dest._00 = (1 - ((yy + zz)))
        dest._10 = (xy - wz)
        dest._20 = (xz + wy)
        dest._30 = 0
        dest._01 = (xy + wz)
        dest._11 = (1 - ((xx + zz)))
        dest._21 = (yz - wx)
        dest._31 = 0
        dest._02 = (xz - wy)
        dest._12 = (yz + wx)
        dest._22 = (1 - ((xx + yy)))
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        return dest

    @staticmethod
    def scale(amount,dest):
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        dest._00 = amount.x
        dest._11 = amount.y
        dest._22 = amount.z
        return dest

    @staticmethod
    def transform(translation,rotation,scale,dest):
        x2 = (rotation.x + rotation.x)
        y2 = (rotation.y + rotation.y)
        z2 = (rotation.z + rotation.z)
        xx = (rotation.x * x2)
        xy = (rotation.x * y2)
        xz = (rotation.x * z2)
        yy = (rotation.y * y2)
        yz = (rotation.y * z2)
        zz = (rotation.z * z2)
        wx = (rotation.w * x2)
        wy = (rotation.w * y2)
        wz = (rotation.w * z2)
        dest._00 = (((1 - ((yy + zz)))) * scale.x)
        dest._01 = (((xy + wz)) * scale.x)
        dest._02 = (((xz - wy)) * scale.x)
        dest._03 = 0
        dest._10 = (((xy - wz)) * scale.y)
        dest._11 = (((1 - ((xx + zz)))) * scale.y)
        dest._12 = (((yz + wx)) * scale.y)
        dest._13 = 0
        dest._20 = (((xz + wy)) * scale.z)
        dest._21 = (((yz - wx)) * scale.z)
        dest._22 = (((1 - ((xx + yy)))) * scale.z)
        dest._23 = 0
        dest._30 = translation.x
        dest._31 = translation.y
        dest._32 = translation.z
        dest._33 = 1
        return dest

    @staticmethod
    def lookAt(eye,centre,up,dest):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest1 = this1
        dest1.x = (centre.x - eye.x)
        dest1.y = (centre.y - eye.y)
        dest1.z = (centre.z - eye.z)
        f = dest1
        v = (((f.x * f.x) + ((f.y * f.y))) + ((f.z * f.z)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        _hx_local_0 = f
        _hx_local_1 = _hx_local_0.x
        _hx_local_0.x = (_hx_local_1 * mult)
        _hx_local_0.x
        _hx_local_2 = f
        _hx_local_3 = _hx_local_2.y
        _hx_local_2.y = (_hx_local_3 * mult)
        _hx_local_2.y
        _hx_local_4 = f
        _hx_local_5 = _hx_local_4.z
        _hx_local_4.z = (_hx_local_5 * mult)
        _hx_local_4.z
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest1 = this1
        x = ((f.y * up.z) - ((f.z * up.y)))
        y = ((f.z * up.x) - ((f.x * up.z)))
        z = ((f.x * up.y) - ((f.y * up.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest1 = this1
        s = dest1
        v = (((s.x * s.x) + ((s.y * s.y))) + ((s.z * s.z)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        _hx_local_6 = s
        _hx_local_7 = _hx_local_6.x
        _hx_local_6.x = (_hx_local_7 * mult)
        _hx_local_6.x
        _hx_local_8 = s
        _hx_local_9 = _hx_local_8.y
        _hx_local_8.y = (_hx_local_9 * mult)
        _hx_local_8.y
        _hx_local_10 = s
        _hx_local_11 = _hx_local_10.z
        _hx_local_10.z = (_hx_local_11 * mult)
        _hx_local_10.z
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest1 = this1
        x = ((s.y * f.z) - ((s.z * f.y)))
        y = ((s.z * f.x) - ((s.x * f.z)))
        z = ((s.x * f.y) - ((s.y * f.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest1 = this1
        u = dest1
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        dest._00 = s.x
        dest._10 = s.y
        dest._20 = s.z
        dest._01 = u.x
        dest._11 = u.y
        dest._21 = u.z
        dest._02 = -f.x
        dest._12 = -f.y
        dest._22 = -f.z
        dest._30 = -((((s.x * eye.x) + ((s.y * eye.y))) + ((s.z * eye.z))))
        dest._31 = -((((u.x * eye.x) + ((u.y * eye.y))) + ((u.z * eye.z))))
        dest._32 = (((f.x * eye.x) + ((f.y * eye.y))) + ((f.z * eye.z)))
        return dest

    @staticmethod
    def perspective(fovy,aspectRatio,near,far,dest):
        f = (1 / Math.tan((fovy / 2)))
        nf = (1 / ((near - far)))
        dest._00 = (f / aspectRatio)
        dest._01 = 0
        dest._02 = 0
        dest._03 = 0
        dest._10 = 0
        dest._11 = f
        dest._12 = 0
        dest._13 = 0
        dest._20 = 0
        dest._21 = 0
        dest._22 = (((far + near)) * nf)
        dest._23 = -1
        dest._30 = 0
        dest._31 = 0
        dest._32 = (((2 * far) * near) * nf)
        dest._33 = 0
        return dest

    @staticmethod
    def orthographic(left,right,bottom,top,near = None,far = None,dest = None):
        if (near is None):
            near = -1
        if (far is None):
            far = 1
        rl = (1 / ((right - left)))
        tb = (1 / ((top - bottom)))
        fn = (1 / ((far - near)))
        dest._00 = (2 * rl)
        dest._10 = 0
        dest._20 = 0
        dest._30 = ((-1 * ((left + right))) * rl)
        dest._01 = 0
        dest._11 = (2 * tb)
        dest._21 = 0
        dest._31 = ((-1 * ((top + bottom))) * tb)
        dest._02 = 0
        dest._12 = 0
        dest._22 = (-2 * fn)
        dest._32 = ((-1 * ((far + near))) * fn)
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        return dest

    @staticmethod
    def frustum(left,right,bottom,top,near = None,far = None,dest = None):
        if (near is None):
            near = -1
        if (far is None):
            far = 1
        rl = (1 / ((right - left)))
        tb = (1 / ((top - bottom)))
        nf = (1 / ((near - far)))
        dest._00 = ((near * 2) * rl)
        dest._01 = 0
        dest._02 = 0
        dest._03 = 0
        dest._10 = 0
        dest._11 = ((near * 2) * tb)
        dest._12 = 0
        dest._13 = 0
        dest._20 = (((right + left)) * tb)
        dest._21 = (((top + bottom)) * tb)
        dest._22 = (((far + near)) * nf)
        dest._23 = -1
        dest._30 = 0
        dest._31 = 0
        dest._32 = (((far * near) * 2) * nf)
        dest._33 = 0
        return dest
glm_GLM._hx_class = glm_GLM


class glm_Mat3Base:
    _hx_class_name = "glm.Mat3Base"
    __slots__ = ("_00", "_10", "_20", "_01", "_11", "_21", "_02", "_12", "_22")
    _hx_fields = ["_00", "_10", "_20", "_01", "_11", "_21", "_02", "_12", "_22"]

    def __init__(self):
        self._22 = None
        self._12 = None
        self._02 = None
        self._21 = None
        self._11 = None
        self._01 = None
        self._20 = None
        self._10 = None
        self._00 = None

glm_Mat3Base._hx_class = glm_Mat3Base


class glm__Mat3_Mat3_Impl_:
    _hx_class_name = "glm._Mat3.Mat3_Impl_"
    __slots__ = ()
    _hx_statics = ["_new", "get_r0c0", "set_r0c0", "get_r1c0", "set_r1c0", "get_r2c0", "set_r2c0", "get_r0c1", "set_r0c1", "get_r1c1", "set_r1c1", "get_r2c1", "set_r2c1", "get_r0c2", "set_r0c2", "get_r1c2", "set_r1c2", "get_r2c2", "set_r2c2", "get", "arrayWrite", "equals", "toString", "identity", "copy", "transpose", "cofactor", "determinant", "invert", "multMat", "multMatOp", "multVec", "multVecOp", "fromFloatArray", "toFloatArray"]
    r0c0 = None
    r1c0 = None
    r2c0 = None
    r0c1 = None
    r1c1 = None
    r2c1 = None
    r0c2 = None
    r1c2 = None
    r2c2 = None

    @staticmethod
    def _new(_r0c0 = None,_r0c1 = None,_r0c2 = None,_r1c0 = None,_r1c1 = None,_r1c2 = None,_r2c0 = None,_r2c1 = None,_r2c2 = None):
        if (_r0c0 is None):
            _r0c0 = 0
        if (_r0c1 is None):
            _r0c1 = 0
        if (_r0c2 is None):
            _r0c2 = 0
        if (_r1c0 is None):
            _r1c0 = 0
        if (_r1c1 is None):
            _r1c1 = 0
        if (_r1c2 is None):
            _r1c2 = 0
        if (_r2c0 is None):
            _r2c0 = 0
        if (_r2c1 is None):
            _r2c1 = 0
        if (_r2c2 is None):
            _r2c2 = 0
        this1 = glm_Mat3Base()
        this1._00 = _r0c0
        this1._01 = _r1c0
        this1._02 = _r2c0
        this1._10 = _r0c1
        this1._11 = _r1c1
        this1._12 = _r2c1
        this1._20 = _r0c2
        this1._21 = _r1c2
        this1._22 = _r2c2
        return this1

    @staticmethod
    def get_r0c0(this1):
        return this1._00

    @staticmethod
    def set_r0c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._00 = v
                return this1._00
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c0(this1):
        return this1._01

    @staticmethod
    def set_r1c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._01 = v
                return this1._01
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c0(this1):
        return this1._02

    @staticmethod
    def set_r2c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._02 = v
                return this1._02
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r0c1(this1):
        return this1._10

    @staticmethod
    def set_r0c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._10 = v
                return this1._10
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c1(this1):
        return this1._11

    @staticmethod
    def set_r1c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._11 = v
                return this1._11
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c1(this1):
        return this1._12

    @staticmethod
    def set_r2c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._12 = v
                return this1._12
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r0c2(this1):
        return this1._20

    @staticmethod
    def set_r0c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._20 = v
                return this1._20
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c2(this1):
        return this1._21

    @staticmethod
    def set_r1c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._21 = v
                return this1._21
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c2(this1):
        return this1._22

    @staticmethod
    def set_r2c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._22 = v
                return this1._22
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1._00
        elif (key1 == 1):
            return this1._01
        elif (key1 == 2):
            return this1._02
        elif (key1 == 3):
            return this1._10
        elif (key1 == 4):
            return this1._11
        elif (key1 == 5):
            return this1._12
        elif (key1 == 6):
            return this1._20
        elif (key1 == 7):
            return this1._21
        elif (key1 == 8):
            return this1._22
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-8)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1._00 = value
                    return this1._00
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1._01 = value
                    return this1._01
                return _hx_local_2()
            return _hx_local_3()
        elif (key1 == 2):
            def _hx_local_5():
                def _hx_local_4():
                    this1._02 = value
                    return this1._02
                return _hx_local_4()
            return _hx_local_5()
        elif (key1 == 3):
            def _hx_local_7():
                def _hx_local_6():
                    this1._10 = value
                    return this1._10
                return _hx_local_6()
            return _hx_local_7()
        elif (key1 == 4):
            def _hx_local_9():
                def _hx_local_8():
                    this1._11 = value
                    return this1._11
                return _hx_local_8()
            return _hx_local_9()
        elif (key1 == 5):
            def _hx_local_11():
                def _hx_local_10():
                    this1._12 = value
                    return this1._12
                return _hx_local_10()
            return _hx_local_11()
        elif (key1 == 6):
            def _hx_local_13():
                def _hx_local_12():
                    this1._20 = value
                    return this1._20
                return _hx_local_12()
            return _hx_local_13()
        elif (key1 == 7):
            def _hx_local_15():
                def _hx_local_14():
                    this1._21 = value
                    return this1._21
                return _hx_local_14()
            return _hx_local_15()
        elif (key1 == 8):
            def _hx_local_17():
                def _hx_local_16():
                    this1._22 = value
                    return this1._22
                return _hx_local_16()
            return _hx_local_17()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-8)!"))

    @staticmethod
    def equals(this1,b):
        return (not ((((((((((Reflect.field(Math,"fabs")((this1._00 - b._00)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1._10 - b._10)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._20 - b._20)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._01 - b._01)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._11 - b._11)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._21 - b._21)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._02 - b._02)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._12 - b._12)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._22 - b._22)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return (((((((("[" + Std.string(this1._00)) + ", ") + Std.string(this1._10)) + ", ") + Std.string(this1._20)) + "]\n") + HxOverrides.stringOrNull(((((((("[" + Std.string(this1._01)) + ", ") + Std.string(this1._11)) + ", ") + Std.string(this1._21)) + "]\n")))) + HxOverrides.stringOrNull(((((((("[" + Std.string(this1._02)) + ", ") + Std.string(this1._12)) + ", ") + Std.string(this1._22)) + "]\n"))))

    @staticmethod
    def identity(dest):
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        return dest

    @staticmethod
    def copy(src,dest):
        dest._00 = src._00
        dest._10 = src._10
        dest._20 = src._20
        dest._01 = src._01
        dest._11 = src._11
        dest._21 = src._21
        dest._02 = src._02
        dest._12 = src._12
        dest._22 = src._22
        return dest

    @staticmethod
    def transpose(src,dest):
        src_r1c0 = src._01
        src_r2c0 = src._02
        src_r2c1 = src._12
        dest._00 = src._00
        dest._01 = src._10
        dest._02 = src._20
        dest._10 = src_r1c0
        dest._11 = src._11
        dest._12 = src._21
        dest._20 = src_r2c0
        dest._21 = src_r2c1
        dest._22 = src._22
        return dest

    @staticmethod
    def cofactor(a,b,c,d):
        return ((a * d) - ((b * c)))

    @staticmethod
    def determinant(src):
        c00 = ((src._11 * src._22) - ((src._21 * src._12)))
        c01 = ((src._10 * src._22) - ((src._20 * src._12)))
        c02 = ((src._10 * src._21) - ((src._20 * src._11)))
        return (((src._00 * c00) - ((src._01 * c01))) + ((src._02 * c02)))

    @staticmethod
    def invert(src,dest):
        c00 = ((src._11 * src._22) - ((src._21 * src._12)))
        c01 = ((src._10 * src._22) - ((src._20 * src._12)))
        c02 = ((src._10 * src._21) - ((src._20 * src._11)))
        det = (((src._00 * c00) - ((src._01 * c01))) + ((src._02 * c02)))
        if (Reflect.field(Math,"fabs")(det) < glm_GLM.EPSILON):
            raise haxe_Exception.thrown("determinant is too small")
        c10 = ((src._01 * src._22) - ((src._21 * src._02)))
        c11 = ((src._00 * src._22) - ((src._20 * src._02)))
        c12 = ((src._00 * src._21) - ((src._20 * src._01)))
        c20 = ((src._01 * src._12) - ((src._11 * src._02)))
        c21 = ((src._00 * src._12) - ((src._10 * src._02)))
        c22 = ((src._00 * src._11) - ((src._10 * src._01)))
        invdet = (1.0 / det)
        dest._00 = (c00 * invdet)
        dest._01 = (-c01 * invdet)
        dest._02 = (c02 * invdet)
        dest._10 = (-c10 * invdet)
        dest._11 = (c11 * invdet)
        dest._12 = (-c12 * invdet)
        dest._20 = (c20 * invdet)
        dest._21 = (-c21 * invdet)
        dest._22 = (c22 * invdet)
        return dest

    @staticmethod
    def multMat(a,b,dest):
        _a = None
        _b = None
        if (dest == a):
            this1 = glm_Mat3Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            dest1 = this1
            dest1._00 = a._00
            dest1._10 = a._10
            dest1._20 = a._20
            dest1._01 = a._01
            dest1._11 = a._11
            dest1._21 = a._21
            dest1._02 = a._02
            dest1._12 = a._12
            dest1._22 = a._22
            _a = dest1
            _b = b
        elif (dest == b):
            _a = a
            this1 = glm_Mat3Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            dest1 = this1
            dest1._00 = b._00
            dest1._10 = b._10
            dest1._20 = b._20
            dest1._01 = b._01
            dest1._11 = b._11
            dest1._21 = b._21
            dest1._02 = b._02
            dest1._12 = b._12
            dest1._22 = b._22
            _b = dest1
        else:
            _a = a
            _b = b
        dest._00 = (((_a._00 * _b._00) + ((_a._10 * _b._01))) + ((_a._20 * _b._02)))
        dest._10 = (((_a._00 * _b._10) + ((_a._10 * _b._11))) + ((_a._20 * _b._12)))
        dest._20 = (((_a._00 * _b._20) + ((_a._10 * _b._21))) + ((_a._20 * _b._22)))
        dest._01 = (((_a._01 * _b._00) + ((_a._11 * _b._01))) + ((_a._21 * _b._02)))
        dest._11 = (((_a._01 * _b._10) + ((_a._11 * _b._11))) + ((_a._21 * _b._12)))
        dest._21 = (((_a._01 * _b._20) + ((_a._11 * _b._21))) + ((_a._21 * _b._22)))
        dest._02 = (((_a._02 * _b._00) + ((_a._12 * _b._01))) + ((_a._22 * _b._02)))
        dest._12 = (((_a._02 * _b._10) + ((_a._12 * _b._11))) + ((_a._22 * _b._12)))
        dest._22 = (((_a._02 * _b._20) + ((_a._12 * _b._21))) + ((_a._22 * _b._22)))
        return dest

    @staticmethod
    def multMatOp(a,b):
        this1 = glm_Mat3Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        dest = this1
        _a = None
        _b = None
        if (dest == a):
            this1 = glm_Mat3Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            dest1 = this1
            dest1._00 = a._00
            dest1._10 = a._10
            dest1._20 = a._20
            dest1._01 = a._01
            dest1._11 = a._11
            dest1._21 = a._21
            dest1._02 = a._02
            dest1._12 = a._12
            dest1._22 = a._22
            _a = dest1
            _b = b
        elif (dest == b):
            _a = a
            this1 = glm_Mat3Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            dest1 = this1
            dest1._00 = b._00
            dest1._10 = b._10
            dest1._20 = b._20
            dest1._01 = b._01
            dest1._11 = b._11
            dest1._21 = b._21
            dest1._02 = b._02
            dest1._12 = b._12
            dest1._22 = b._22
            _b = dest1
        else:
            _a = a
            _b = b
        dest._00 = (((_a._00 * _b._00) + ((_a._10 * _b._01))) + ((_a._20 * _b._02)))
        dest._10 = (((_a._00 * _b._10) + ((_a._10 * _b._11))) + ((_a._20 * _b._12)))
        dest._20 = (((_a._00 * _b._20) + ((_a._10 * _b._21))) + ((_a._20 * _b._22)))
        dest._01 = (((_a._01 * _b._00) + ((_a._11 * _b._01))) + ((_a._21 * _b._02)))
        dest._11 = (((_a._01 * _b._10) + ((_a._11 * _b._11))) + ((_a._21 * _b._12)))
        dest._21 = (((_a._01 * _b._20) + ((_a._11 * _b._21))) + ((_a._21 * _b._22)))
        dest._02 = (((_a._02 * _b._00) + ((_a._12 * _b._01))) + ((_a._22 * _b._02)))
        dest._12 = (((_a._02 * _b._10) + ((_a._12 * _b._11))) + ((_a._22 * _b._12)))
        dest._22 = (((_a._02 * _b._20) + ((_a._12 * _b._21))) + ((_a._22 * _b._22)))
        return dest

    @staticmethod
    def multVec(m,v,dest):
        x = v.x
        y = v.y
        z = v.z
        dest.x = (((m._00 * x) + ((m._10 * y))) + ((m._20 * z)))
        dest.y = (((m._01 * x) + ((m._11 * y))) + ((m._21 * z)))
        dest.z = (((m._02 * x) + ((m._12 * y))) + ((m._22 * z)))
        return dest

    @staticmethod
    def multVecOp(m,v):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        x = v.x
        y = v.y
        z = v.z
        dest.x = (((m._00 * x) + ((m._10 * y))) + ((m._20 * z)))
        dest.y = (((m._01 * x) + ((m._11 * y))) + ((m._21 * z)))
        dest.z = (((m._02 * x) + ((m._12 * y))) + ((m._22 * z)))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        _r0c0 = (arr[0] if 0 < len(arr) else None)
        _r0c1 = (arr[3] if 3 < len(arr) else None)
        _r0c2 = (arr[6] if 6 < len(arr) else None)
        _r1c0 = (arr[1] if 1 < len(arr) else None)
        _r1c1 = (arr[4] if 4 < len(arr) else None)
        _r1c2 = (arr[7] if 7 < len(arr) else None)
        _r2c0 = (arr[2] if 2 < len(arr) else None)
        _r2c1 = (arr[5] if 5 < len(arr) else None)
        _r2c2 = (arr[8] if 8 < len(arr) else None)
        if (_r2c2 is None):
            _r2c2 = 0
        if (_r2c1 is None):
            _r2c1 = 0
        if (_r2c0 is None):
            _r2c0 = 0
        if (_r1c2 is None):
            _r1c2 = 0
        if (_r1c1 is None):
            _r1c1 = 0
        if (_r1c0 is None):
            _r1c0 = 0
        if (_r0c2 is None):
            _r0c2 = 0
        if (_r0c1 is None):
            _r0c1 = 0
        if (_r0c0 is None):
            _r0c0 = 0
        this1 = glm_Mat3Base()
        this1._00 = _r0c0
        this1._01 = _r1c0
        this1._02 = _r2c0
        this1._10 = _r0c1
        this1._11 = _r1c1
        this1._12 = _r2c1
        this1._20 = _r0c2
        this1._21 = _r1c2
        this1._22 = _r2c2
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1._00, this1._01, this1._02, this1._10, this1._11, this1._12, this1._20, this1._21, this1._22]
glm__Mat3_Mat3_Impl_._hx_class = glm__Mat3_Mat3_Impl_


class glm_Mat4Base:
    _hx_class_name = "glm.Mat4Base"
    __slots__ = ("_00", "_10", "_20", "_30", "_01", "_11", "_21", "_31", "_02", "_12", "_22", "_32", "_03", "_13", "_23", "_33")
    _hx_fields = ["_00", "_10", "_20", "_30", "_01", "_11", "_21", "_31", "_02", "_12", "_22", "_32", "_03", "_13", "_23", "_33"]

    def __init__(self):
        self._33 = None
        self._23 = None
        self._13 = None
        self._03 = None
        self._32 = None
        self._22 = None
        self._12 = None
        self._02 = None
        self._31 = None
        self._21 = None
        self._11 = None
        self._01 = None
        self._30 = None
        self._20 = None
        self._10 = None
        self._00 = None

glm_Mat4Base._hx_class = glm_Mat4Base


class glm__Mat4_Mat4_Impl_:
    _hx_class_name = "glm._Mat4.Mat4_Impl_"
    __slots__ = ()
    _hx_statics = ["_new", "get_r0c0", "set_r0c0", "get_r1c0", "set_r1c0", "get_r2c0", "set_r2c0", "get_r3c0", "set_r3c0", "get_r0c1", "set_r0c1", "get_r1c1", "set_r1c1", "get_r2c1", "set_r2c1", "get_r3c1", "set_r3c1", "get_r0c2", "set_r0c2", "get_r1c2", "set_r1c2", "get_r2c2", "set_r2c2", "get_r3c2", "set_r3c2", "get_r0c3", "set_r0c3", "get_r1c3", "set_r1c3", "get_r2c3", "set_r2c3", "get_r3c3", "set_r3c3", "get", "arrayWrite", "equals", "toString", "identity", "copy", "transpose", "determinant", "invert", "multMat", "multMatOp", "multVec", "multVecOp", "fromFloatArray", "toFloatArray"]
    r0c0 = None
    r1c0 = None
    r2c0 = None
    r3c0 = None
    r0c1 = None
    r1c1 = None
    r2c1 = None
    r3c1 = None
    r0c2 = None
    r1c2 = None
    r2c2 = None
    r3c2 = None
    r0c3 = None
    r1c3 = None
    r2c3 = None
    r3c3 = None

    @staticmethod
    def _new(_r0c0 = None,_r0c1 = None,_r0c2 = None,_r0c3 = None,_r1c0 = None,_r1c1 = None,_r1c2 = None,_r1c3 = None,_r2c0 = None,_r2c1 = None,_r2c2 = None,_r2c3 = None,_r3c0 = None,_r3c1 = None,_r3c2 = None,_r3c3 = None):
        if (_r0c0 is None):
            _r0c0 = 0
        if (_r0c1 is None):
            _r0c1 = 0
        if (_r0c2 is None):
            _r0c2 = 0
        if (_r0c3 is None):
            _r0c3 = 0
        if (_r1c0 is None):
            _r1c0 = 0
        if (_r1c1 is None):
            _r1c1 = 0
        if (_r1c2 is None):
            _r1c2 = 0
        if (_r1c3 is None):
            _r1c3 = 0
        if (_r2c0 is None):
            _r2c0 = 0
        if (_r2c1 is None):
            _r2c1 = 0
        if (_r2c2 is None):
            _r2c2 = 0
        if (_r2c3 is None):
            _r2c3 = 0
        if (_r3c0 is None):
            _r3c0 = 0
        if (_r3c1 is None):
            _r3c1 = 0
        if (_r3c2 is None):
            _r3c2 = 0
        if (_r3c3 is None):
            _r3c3 = 0
        this1 = glm_Mat4Base()
        this1._00 = _r0c0
        this1._01 = _r1c0
        this1._02 = _r2c0
        this1._03 = _r3c0
        this1._10 = _r0c1
        this1._11 = _r1c1
        this1._12 = _r2c1
        this1._13 = _r3c1
        this1._20 = _r0c2
        this1._21 = _r1c2
        this1._22 = _r2c2
        this1._23 = _r3c2
        this1._30 = _r0c3
        this1._31 = _r1c3
        this1._32 = _r2c3
        this1._33 = _r3c3
        return this1

    @staticmethod
    def get_r0c0(this1):
        return this1._00

    @staticmethod
    def set_r0c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._00 = v
                return this1._00
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c0(this1):
        return this1._01

    @staticmethod
    def set_r1c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._01 = v
                return this1._01
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c0(this1):
        return this1._02

    @staticmethod
    def set_r2c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._02 = v
                return this1._02
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r3c0(this1):
        return this1._03

    @staticmethod
    def set_r3c0(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._03 = v
                return this1._03
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r0c1(this1):
        return this1._10

    @staticmethod
    def set_r0c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._10 = v
                return this1._10
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c1(this1):
        return this1._11

    @staticmethod
    def set_r1c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._11 = v
                return this1._11
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c1(this1):
        return this1._12

    @staticmethod
    def set_r2c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._12 = v
                return this1._12
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r3c1(this1):
        return this1._13

    @staticmethod
    def set_r3c1(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._13 = v
                return this1._13
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r0c2(this1):
        return this1._20

    @staticmethod
    def set_r0c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._20 = v
                return this1._20
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c2(this1):
        return this1._21

    @staticmethod
    def set_r1c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._21 = v
                return this1._21
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c2(this1):
        return this1._22

    @staticmethod
    def set_r2c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._22 = v
                return this1._22
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r3c2(this1):
        return this1._23

    @staticmethod
    def set_r3c2(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._23 = v
                return this1._23
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r0c3(this1):
        return this1._30

    @staticmethod
    def set_r0c3(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._30 = v
                return this1._30
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r1c3(this1):
        return this1._31

    @staticmethod
    def set_r1c3(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._31 = v
                return this1._31
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r2c3(this1):
        return this1._32

    @staticmethod
    def set_r2c3(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._32 = v
                return this1._32
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r3c3(this1):
        return this1._33

    @staticmethod
    def set_r3c3(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1._33 = v
                return this1._33
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1._00
        elif (key1 == 1):
            return this1._01
        elif (key1 == 2):
            return this1._02
        elif (key1 == 3):
            return this1._03
        elif (key1 == 4):
            return this1._10
        elif (key1 == 5):
            return this1._11
        elif (key1 == 6):
            return this1._12
        elif (key1 == 7):
            return this1._13
        elif (key1 == 8):
            return this1._20
        elif (key1 == 9):
            return this1._21
        elif (key1 == 10):
            return this1._22
        elif (key1 == 11):
            return this1._23
        elif (key1 == 12):
            return this1._30
        elif (key1 == 13):
            return this1._31
        elif (key1 == 14):
            return this1._32
        elif (key1 == 15):
            return this1._33
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-15)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1._00 = value
                    return this1._00
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1._01 = value
                    return this1._01
                return _hx_local_2()
            return _hx_local_3()
        elif (key1 == 2):
            def _hx_local_5():
                def _hx_local_4():
                    this1._02 = value
                    return this1._02
                return _hx_local_4()
            return _hx_local_5()
        elif (key1 == 3):
            def _hx_local_7():
                def _hx_local_6():
                    this1._03 = value
                    return this1._03
                return _hx_local_6()
            return _hx_local_7()
        elif (key1 == 4):
            def _hx_local_9():
                def _hx_local_8():
                    this1._10 = value
                    return this1._10
                return _hx_local_8()
            return _hx_local_9()
        elif (key1 == 5):
            def _hx_local_11():
                def _hx_local_10():
                    this1._11 = value
                    return this1._11
                return _hx_local_10()
            return _hx_local_11()
        elif (key1 == 6):
            def _hx_local_13():
                def _hx_local_12():
                    this1._12 = value
                    return this1._12
                return _hx_local_12()
            return _hx_local_13()
        elif (key1 == 7):
            def _hx_local_15():
                def _hx_local_14():
                    this1._13 = value
                    return this1._13
                return _hx_local_14()
            return _hx_local_15()
        elif (key1 == 8):
            def _hx_local_17():
                def _hx_local_16():
                    this1._20 = value
                    return this1._20
                return _hx_local_16()
            return _hx_local_17()
        elif (key1 == 9):
            def _hx_local_19():
                def _hx_local_18():
                    this1._21 = value
                    return this1._21
                return _hx_local_18()
            return _hx_local_19()
        elif (key1 == 10):
            def _hx_local_21():
                def _hx_local_20():
                    this1._22 = value
                    return this1._22
                return _hx_local_20()
            return _hx_local_21()
        elif (key1 == 11):
            def _hx_local_23():
                def _hx_local_22():
                    this1._23 = value
                    return this1._23
                return _hx_local_22()
            return _hx_local_23()
        elif (key1 == 12):
            def _hx_local_25():
                def _hx_local_24():
                    this1._30 = value
                    return this1._30
                return _hx_local_24()
            return _hx_local_25()
        elif (key1 == 13):
            def _hx_local_27():
                def _hx_local_26():
                    this1._31 = value
                    return this1._31
                return _hx_local_26()
            return _hx_local_27()
        elif (key1 == 14):
            def _hx_local_29():
                def _hx_local_28():
                    this1._32 = value
                    return this1._32
                return _hx_local_28()
            return _hx_local_29()
        elif (key1 == 15):
            def _hx_local_31():
                def _hx_local_30():
                    this1._33 = value
                    return this1._33
                return _hx_local_30()
            return _hx_local_31()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-15)!"))

    @staticmethod
    def equals(this1,b):
        return (not (((((((((((((((((Reflect.field(Math,"fabs")((this1._00 - b._00)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1._10 - b._10)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._20 - b._20)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._30 - b._30)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._01 - b._01)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._11 - b._11)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._21 - b._21)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._31 - b._31)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._02 - b._02)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._12 - b._12)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._22 - b._22)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._32 - b._32)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._03 - b._03)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._13 - b._13)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._23 - b._23)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1._33 - b._33)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return ((((((((((("[" + Std.string(this1._00)) + ", ") + Std.string(this1._10)) + ", ") + Std.string(this1._20)) + ", ") + Std.string(this1._30)) + "]\n") + HxOverrides.stringOrNull(((((((((("[" + Std.string(this1._01)) + ", ") + Std.string(this1._11)) + ", ") + Std.string(this1._21)) + ", ") + Std.string(this1._31)) + "]\n")))) + HxOverrides.stringOrNull(((((((((("[" + Std.string(this1._02)) + ", ") + Std.string(this1._12)) + ", ") + Std.string(this1._22)) + ", ") + Std.string(this1._32)) + "]\n")))) + HxOverrides.stringOrNull(((((((((("[" + Std.string(this1._03)) + ", ") + Std.string(this1._13)) + ", ") + Std.string(this1._23)) + ", ") + Std.string(this1._33)) + "]\n"))))

    @staticmethod
    def identity(dest):
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        return dest

    @staticmethod
    def copy(src,dest):
        dest._00 = src._00
        dest._10 = src._10
        dest._20 = src._20
        dest._30 = src._30
        dest._01 = src._01
        dest._11 = src._11
        dest._21 = src._21
        dest._31 = src._31
        dest._02 = src._02
        dest._12 = src._12
        dest._22 = src._22
        dest._32 = src._32
        dest._03 = src._03
        dest._13 = src._13
        dest._23 = src._23
        dest._33 = src._33
        return dest

    @staticmethod
    def transpose(src,dest):
        src_r1c0 = src._01
        src_r2c0 = src._02
        src_r2c1 = src._12
        src_r3c0 = src._03
        src_r3c1 = src._13
        src_r3c2 = src._23
        dest._00 = src._00
        dest._01 = src._10
        dest._02 = src._20
        dest._03 = src._30
        dest._10 = src_r1c0
        dest._11 = src._11
        dest._12 = src._21
        dest._13 = src._31
        dest._20 = src_r2c0
        dest._21 = src_r2c1
        dest._22 = src._22
        dest._23 = src._32
        dest._30 = src_r3c0
        dest._31 = src_r3c1
        dest._32 = src_r3c2
        dest._33 = src._33
        return dest

    @staticmethod
    def determinant(src):
        a00 = src._00
        a01 = src._01
        a02 = src._02
        a03 = src._03
        a10 = src._10
        a11 = src._11
        a12 = src._12
        a13 = src._13
        a20 = src._20
        a21 = src._21
        a22 = src._22
        a23 = src._23
        a30 = src._30
        a31 = src._31
        a32 = src._32
        a33 = src._33
        b00 = ((a00 * a11) - ((a01 * a10)))
        b01 = ((a00 * a12) - ((a02 * a10)))
        b02 = ((a00 * a13) - ((a03 * a10)))
        b03 = ((a01 * a12) - ((a02 * a11)))
        b04 = ((a01 * a13) - ((a03 * a11)))
        b05 = ((a02 * a13) - ((a03 * a12)))
        b06 = ((a20 * a31) - ((a21 * a30)))
        b07 = ((a20 * a32) - ((a22 * a30)))
        b08 = ((a20 * a33) - ((a23 * a30)))
        b09 = ((a21 * a32) - ((a22 * a31)))
        b10 = ((a21 * a33) - ((a23 * a31)))
        b11 = ((a22 * a33) - ((a23 * a32)))
        return ((((((b00 * b11) - ((b01 * b10))) + ((b02 * b09))) + ((b03 * b08))) - ((b04 * b07))) + ((b05 * b06)))

    @staticmethod
    def invert(src,dest):
        a00 = src._00
        a01 = src._10
        a02 = src._20
        a03 = src._30
        a10 = src._01
        a11 = src._11
        a12 = src._21
        a13 = src._31
        a20 = src._02
        a21 = src._12
        a22 = src._22
        a23 = src._32
        a30 = src._03
        a31 = src._13
        a32 = src._23
        a33 = src._33
        t00 = (((((((a12 * a23) * a31) - (((a13 * a22) * a31))) + (((a13 * a21) * a32))) - (((a11 * a23) * a32))) - (((a12 * a21) * a33))) + (((a11 * a22) * a33)))
        t01 = (((((((a03 * a22) * a31) - (((a02 * a23) * a31))) - (((a03 * a21) * a32))) + (((a01 * a23) * a32))) + (((a02 * a21) * a33))) - (((a01 * a22) * a33)))
        t02 = (((((((a02 * a13) * a31) - (((a03 * a12) * a31))) + (((a03 * a11) * a32))) - (((a01 * a13) * a32))) - (((a02 * a11) * a33))) + (((a01 * a12) * a33)))
        t03 = (((((((a03 * a12) * a21) - (((a02 * a13) * a21))) - (((a03 * a11) * a22))) + (((a01 * a13) * a22))) + (((a02 * a11) * a23))) - (((a01 * a12) * a23)))
        det = ((((a00 * t00) + ((a10 * t01))) + ((a20 * t02))) + ((a30 * t03)))
        if (Reflect.field(Math,"fabs")(det) <= glm_GLM.EPSILON):
            raise haxe_Exception.thrown((("Can't invert matrix, det (" + Std.string(det)) + ") is too small!"))
        idet = (1 / det)
        dest._00 = (t00 * idet)
        dest._01 = (((((((((a13 * a22) * a30) - (((a12 * a23) * a30))) - (((a13 * a20) * a32))) + (((a10 * a23) * a32))) + (((a12 * a20) * a33))) - (((a10 * a22) * a33)))) * idet)
        dest._02 = (((((((((a11 * a23) * a30) - (((a13 * a21) * a30))) + (((a13 * a20) * a31))) - (((a10 * a23) * a31))) - (((a11 * a20) * a33))) + (((a10 * a21) * a33)))) * idet)
        dest._03 = (((((((((a12 * a21) * a30) - (((a11 * a22) * a30))) - (((a12 * a20) * a31))) + (((a10 * a22) * a31))) + (((a11 * a20) * a32))) - (((a10 * a21) * a32)))) * idet)
        dest._10 = (t01 * idet)
        dest._11 = (((((((((a02 * a23) * a30) - (((a03 * a22) * a30))) + (((a03 * a20) * a32))) - (((a00 * a23) * a32))) - (((a02 * a20) * a33))) + (((a00 * a22) * a33)))) * idet)
        dest._12 = (((((((((a03 * a21) * a30) - (((a01 * a23) * a30))) - (((a03 * a20) * a31))) + (((a00 * a23) * a31))) + (((a01 * a20) * a33))) - (((a00 * a21) * a33)))) * idet)
        dest._13 = (((((((((a01 * a22) * a30) - (((a02 * a21) * a30))) + (((a02 * a20) * a31))) - (((a00 * a22) * a31))) - (((a01 * a20) * a32))) + (((a00 * a21) * a32)))) * idet)
        dest._20 = (t02 * idet)
        dest._21 = (((((((((a03 * a12) * a30) - (((a02 * a13) * a30))) - (((a03 * a10) * a32))) + (((a00 * a13) * a32))) + (((a02 * a10) * a33))) - (((a00 * a12) * a33)))) * idet)
        dest._22 = (((((((((a01 * a13) * a30) - (((a03 * a11) * a30))) + (((a03 * a10) * a31))) - (((a00 * a13) * a31))) - (((a01 * a10) * a33))) + (((a00 * a11) * a33)))) * idet)
        dest._23 = (((((((((a02 * a11) * a30) - (((a01 * a12) * a30))) - (((a02 * a10) * a31))) + (((a00 * a12) * a31))) + (((a01 * a10) * a32))) - (((a00 * a11) * a32)))) * idet)
        dest._30 = (t03 * idet)
        dest._31 = (((((((((a02 * a13) * a20) - (((a03 * a12) * a20))) + (((a03 * a10) * a22))) - (((a00 * a13) * a22))) - (((a02 * a10) * a23))) + (((a00 * a12) * a23)))) * idet)
        dest._32 = (((((((((a03 * a11) * a20) - (((a01 * a13) * a20))) - (((a03 * a10) * a21))) + (((a00 * a13) * a21))) + (((a01 * a10) * a23))) - (((a00 * a11) * a23)))) * idet)
        dest._33 = (((((((((a01 * a12) * a20) - (((a02 * a11) * a20))) + (((a02 * a10) * a21))) - (((a00 * a12) * a21))) - (((a01 * a10) * a22))) + (((a00 * a11) * a22)))) * idet)
        return dest

    @staticmethod
    def multMat(a,b,dest):
        _a = None
        _b = None
        if (dest == a):
            this1 = glm_Mat4Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._03 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._13 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            this1._23 = 0
            this1._30 = 0
            this1._31 = 0
            this1._32 = 0
            this1._33 = 0
            dest1 = this1
            dest1._00 = a._00
            dest1._10 = a._10
            dest1._20 = a._20
            dest1._30 = a._30
            dest1._01 = a._01
            dest1._11 = a._11
            dest1._21 = a._21
            dest1._31 = a._31
            dest1._02 = a._02
            dest1._12 = a._12
            dest1._22 = a._22
            dest1._32 = a._32
            dest1._03 = a._03
            dest1._13 = a._13
            dest1._23 = a._23
            dest1._33 = a._33
            _a = dest1
            _b = b
        elif (dest == b):
            _a = a
            this1 = glm_Mat4Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._03 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._13 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            this1._23 = 0
            this1._30 = 0
            this1._31 = 0
            this1._32 = 0
            this1._33 = 0
            dest1 = this1
            dest1._00 = b._00
            dest1._10 = b._10
            dest1._20 = b._20
            dest1._30 = b._30
            dest1._01 = b._01
            dest1._11 = b._11
            dest1._21 = b._21
            dest1._31 = b._31
            dest1._02 = b._02
            dest1._12 = b._12
            dest1._22 = b._22
            dest1._32 = b._32
            dest1._03 = b._03
            dest1._13 = b._13
            dest1._23 = b._23
            dest1._33 = b._33
            _b = dest1
        else:
            _a = a
            _b = b
        dest._00 = ((((_a._00 * _b._00) + ((_a._10 * _b._01))) + ((_a._20 * _b._02))) + ((_a._30 * _b._03)))
        dest._10 = ((((_a._00 * _b._10) + ((_a._10 * _b._11))) + ((_a._20 * _b._12))) + ((_a._30 * _b._13)))
        dest._20 = ((((_a._00 * _b._20) + ((_a._10 * _b._21))) + ((_a._20 * _b._22))) + ((_a._30 * _b._23)))
        dest._30 = ((((_a._00 * _b._30) + ((_a._10 * _b._31))) + ((_a._20 * _b._32))) + ((_a._30 * _b._33)))
        dest._01 = ((((_a._01 * _b._00) + ((_a._11 * _b._01))) + ((_a._21 * _b._02))) + ((_a._31 * _b._03)))
        dest._11 = ((((_a._01 * _b._10) + ((_a._11 * _b._11))) + ((_a._21 * _b._12))) + ((_a._31 * _b._13)))
        dest._21 = ((((_a._01 * _b._20) + ((_a._11 * _b._21))) + ((_a._21 * _b._22))) + ((_a._31 * _b._23)))
        dest._31 = ((((_a._01 * _b._30) + ((_a._11 * _b._31))) + ((_a._21 * _b._32))) + ((_a._31 * _b._33)))
        dest._02 = ((((_a._02 * _b._00) + ((_a._12 * _b._01))) + ((_a._22 * _b._02))) + ((_a._32 * _b._03)))
        dest._12 = ((((_a._02 * _b._10) + ((_a._12 * _b._11))) + ((_a._22 * _b._12))) + ((_a._32 * _b._13)))
        dest._22 = ((((_a._02 * _b._20) + ((_a._12 * _b._21))) + ((_a._22 * _b._22))) + ((_a._32 * _b._23)))
        dest._32 = ((((_a._02 * _b._30) + ((_a._12 * _b._31))) + ((_a._22 * _b._32))) + ((_a._32 * _b._33)))
        dest._03 = ((((_a._03 * _b._00) + ((_a._13 * _b._01))) + ((_a._23 * _b._02))) + ((_a._33 * _b._03)))
        dest._13 = ((((_a._03 * _b._10) + ((_a._13 * _b._11))) + ((_a._23 * _b._12))) + ((_a._33 * _b._13)))
        dest._23 = ((((_a._03 * _b._20) + ((_a._13 * _b._21))) + ((_a._23 * _b._22))) + ((_a._33 * _b._23)))
        dest._33 = ((((_a._03 * _b._30) + ((_a._13 * _b._31))) + ((_a._23 * _b._32))) + ((_a._33 * _b._33)))
        return dest

    @staticmethod
    def multMatOp(a,b):
        this1 = glm_Mat4Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._03 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._13 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        this1._23 = 0
        this1._30 = 0
        this1._31 = 0
        this1._32 = 0
        this1._33 = 0
        dest = this1
        _a = None
        _b = None
        if (dest == a):
            this1 = glm_Mat4Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._03 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._13 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            this1._23 = 0
            this1._30 = 0
            this1._31 = 0
            this1._32 = 0
            this1._33 = 0
            dest1 = this1
            dest1._00 = a._00
            dest1._10 = a._10
            dest1._20 = a._20
            dest1._30 = a._30
            dest1._01 = a._01
            dest1._11 = a._11
            dest1._21 = a._21
            dest1._31 = a._31
            dest1._02 = a._02
            dest1._12 = a._12
            dest1._22 = a._22
            dest1._32 = a._32
            dest1._03 = a._03
            dest1._13 = a._13
            dest1._23 = a._23
            dest1._33 = a._33
            _a = dest1
            _b = b
        elif (dest == b):
            _a = a
            this1 = glm_Mat4Base()
            this1._00 = 0
            this1._01 = 0
            this1._02 = 0
            this1._03 = 0
            this1._10 = 0
            this1._11 = 0
            this1._12 = 0
            this1._13 = 0
            this1._20 = 0
            this1._21 = 0
            this1._22 = 0
            this1._23 = 0
            this1._30 = 0
            this1._31 = 0
            this1._32 = 0
            this1._33 = 0
            dest1 = this1
            dest1._00 = b._00
            dest1._10 = b._10
            dest1._20 = b._20
            dest1._30 = b._30
            dest1._01 = b._01
            dest1._11 = b._11
            dest1._21 = b._21
            dest1._31 = b._31
            dest1._02 = b._02
            dest1._12 = b._12
            dest1._22 = b._22
            dest1._32 = b._32
            dest1._03 = b._03
            dest1._13 = b._13
            dest1._23 = b._23
            dest1._33 = b._33
            _b = dest1
        else:
            _a = a
            _b = b
        dest._00 = ((((_a._00 * _b._00) + ((_a._10 * _b._01))) + ((_a._20 * _b._02))) + ((_a._30 * _b._03)))
        dest._10 = ((((_a._00 * _b._10) + ((_a._10 * _b._11))) + ((_a._20 * _b._12))) + ((_a._30 * _b._13)))
        dest._20 = ((((_a._00 * _b._20) + ((_a._10 * _b._21))) + ((_a._20 * _b._22))) + ((_a._30 * _b._23)))
        dest._30 = ((((_a._00 * _b._30) + ((_a._10 * _b._31))) + ((_a._20 * _b._32))) + ((_a._30 * _b._33)))
        dest._01 = ((((_a._01 * _b._00) + ((_a._11 * _b._01))) + ((_a._21 * _b._02))) + ((_a._31 * _b._03)))
        dest._11 = ((((_a._01 * _b._10) + ((_a._11 * _b._11))) + ((_a._21 * _b._12))) + ((_a._31 * _b._13)))
        dest._21 = ((((_a._01 * _b._20) + ((_a._11 * _b._21))) + ((_a._21 * _b._22))) + ((_a._31 * _b._23)))
        dest._31 = ((((_a._01 * _b._30) + ((_a._11 * _b._31))) + ((_a._21 * _b._32))) + ((_a._31 * _b._33)))
        dest._02 = ((((_a._02 * _b._00) + ((_a._12 * _b._01))) + ((_a._22 * _b._02))) + ((_a._32 * _b._03)))
        dest._12 = ((((_a._02 * _b._10) + ((_a._12 * _b._11))) + ((_a._22 * _b._12))) + ((_a._32 * _b._13)))
        dest._22 = ((((_a._02 * _b._20) + ((_a._12 * _b._21))) + ((_a._22 * _b._22))) + ((_a._32 * _b._23)))
        dest._32 = ((((_a._02 * _b._30) + ((_a._12 * _b._31))) + ((_a._22 * _b._32))) + ((_a._32 * _b._33)))
        dest._03 = ((((_a._03 * _b._00) + ((_a._13 * _b._01))) + ((_a._23 * _b._02))) + ((_a._33 * _b._03)))
        dest._13 = ((((_a._03 * _b._10) + ((_a._13 * _b._11))) + ((_a._23 * _b._12))) + ((_a._33 * _b._13)))
        dest._23 = ((((_a._03 * _b._20) + ((_a._13 * _b._21))) + ((_a._23 * _b._22))) + ((_a._33 * _b._23)))
        dest._33 = ((((_a._03 * _b._30) + ((_a._13 * _b._31))) + ((_a._23 * _b._32))) + ((_a._33 * _b._33)))
        return dest

    @staticmethod
    def multVec(m,v,dest):
        x = v.x
        y = v.y
        z = v.z
        w = v.w
        dest.x = ((((m._00 * x) + ((m._10 * y))) + ((m._20 * z))) + ((m._30 * w)))
        dest.y = ((((m._01 * x) + ((m._11 * y))) + ((m._21 * z))) + ((m._31 * w)))
        dest.z = ((((m._02 * x) + ((m._12 * y))) + ((m._22 * z))) + ((m._32 * w)))
        dest.w = ((((m._03 * x) + ((m._13 * y))) + ((m._23 * z))) + ((m._33 * w)))
        return dest

    @staticmethod
    def multVecOp(m,v):
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        x = v.x
        y = v.y
        z = v.z
        w = v.w
        dest.x = ((((m._00 * x) + ((m._10 * y))) + ((m._20 * z))) + ((m._30 * w)))
        dest.y = ((((m._01 * x) + ((m._11 * y))) + ((m._21 * z))) + ((m._31 * w)))
        dest.z = ((((m._02 * x) + ((m._12 * y))) + ((m._22 * z))) + ((m._32 * w)))
        dest.w = ((((m._03 * x) + ((m._13 * y))) + ((m._23 * z))) + ((m._33 * w)))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        _r0c0 = (arr[0] if 0 < len(arr) else None)
        _r0c1 = (arr[4] if 4 < len(arr) else None)
        _r0c2 = (arr[8] if 8 < len(arr) else None)
        _r0c3 = (arr[12] if 12 < len(arr) else None)
        _r1c0 = (arr[1] if 1 < len(arr) else None)
        _r1c1 = (arr[5] if 5 < len(arr) else None)
        _r1c2 = (arr[9] if 9 < len(arr) else None)
        _r1c3 = (arr[13] if 13 < len(arr) else None)
        _r2c0 = (arr[2] if 2 < len(arr) else None)
        _r2c1 = (arr[6] if 6 < len(arr) else None)
        _r2c2 = (arr[10] if 10 < len(arr) else None)
        _r2c3 = (arr[14] if 14 < len(arr) else None)
        _r3c0 = (arr[3] if 3 < len(arr) else None)
        _r3c1 = (arr[7] if 7 < len(arr) else None)
        _r3c2 = (arr[11] if 11 < len(arr) else None)
        _r3c3 = (arr[15] if 15 < len(arr) else None)
        if (_r3c3 is None):
            _r3c3 = 0
        if (_r3c2 is None):
            _r3c2 = 0
        if (_r3c1 is None):
            _r3c1 = 0
        if (_r3c0 is None):
            _r3c0 = 0
        if (_r2c3 is None):
            _r2c3 = 0
        if (_r2c2 is None):
            _r2c2 = 0
        if (_r2c1 is None):
            _r2c1 = 0
        if (_r2c0 is None):
            _r2c0 = 0
        if (_r1c3 is None):
            _r1c3 = 0
        if (_r1c2 is None):
            _r1c2 = 0
        if (_r1c1 is None):
            _r1c1 = 0
        if (_r1c0 is None):
            _r1c0 = 0
        if (_r0c3 is None):
            _r0c3 = 0
        if (_r0c2 is None):
            _r0c2 = 0
        if (_r0c1 is None):
            _r0c1 = 0
        if (_r0c0 is None):
            _r0c0 = 0
        this1 = glm_Mat4Base()
        this1._00 = _r0c0
        this1._01 = _r1c0
        this1._02 = _r2c0
        this1._03 = _r3c0
        this1._10 = _r0c1
        this1._11 = _r1c1
        this1._12 = _r2c1
        this1._13 = _r3c1
        this1._20 = _r0c2
        this1._21 = _r1c2
        this1._22 = _r2c2
        this1._23 = _r3c2
        this1._30 = _r0c3
        this1._31 = _r1c3
        this1._32 = _r2c3
        this1._33 = _r3c3
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1._00, this1._01, this1._02, this1._03, this1._10, this1._11, this1._12, this1._13, this1._20, this1._21, this1._22, this1._23, this1._30, this1._31, this1._32, this1._33]
glm__Mat4_Mat4_Impl_._hx_class = glm__Mat4_Mat4_Impl_


class glm_QuatBase:
    _hx_class_name = "glm.QuatBase"
    __slots__ = ("x", "y", "z", "w")
    _hx_fields = ["x", "y", "z", "w"]

    def __init__(self):
        self.w = None
        self.z = None
        self.y = None
        self.x = None

glm_QuatBase._hx_class = glm_QuatBase


class glm__Quat_Quat_Impl_:
    _hx_class_name = "glm._Quat.Quat_Impl_"
    __slots__ = ()
    _hx_statics = ["get_x", "set_x", "get_y", "set_y", "get_z", "set_z", "get_w", "set_w", "get", "arrayWrite", "_new", "equals", "toString", "lengthSquared", "length", "normalize", "dot", "identity", "copy", "axisAngle", "multiplyQuats", "multiplyQuatsOp", "multiplyScalar", "multiplyScalarOp", "lerp", "slerp", "invert", "conjugate", "fromEuler", "fromFloatArray", "toFloatArray"]
    x = None
    y = None
    z = None
    w = None

    @staticmethod
    def get_x(this1):
        return this1.x

    @staticmethod
    def set_x(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_y(this1):
        return this1.y

    @staticmethod
    def set_y(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_z(this1):
        return this1.z

    @staticmethod
    def set_z(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.z = v
                return this1.z
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_w(this1):
        return this1.w

    @staticmethod
    def set_w(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.w = v
                return this1.w
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1.x
        elif (key1 == 1):
            return this1.y
        elif (key1 == 2):
            return this1.z
        elif (key1 == 3):
            return this1.w
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-3)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1.x = value
                    return this1.x
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1.y = value
                    return this1.y
                return _hx_local_2()
            return _hx_local_3()
        elif (key1 == 2):
            def _hx_local_5():
                def _hx_local_4():
                    this1.z = value
                    return this1.z
                return _hx_local_4()
            return _hx_local_5()
        elif (key1 == 3):
            def _hx_local_7():
                def _hx_local_6():
                    this1.w = value
                    return this1.w
                return _hx_local_6()
            return _hx_local_7()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-3)!"))

    @staticmethod
    def _new(x = None,y = None,z = None,w = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        if (z is None):
            z = 0
        if (w is None):
            w = 1
        this1 = glm_QuatBase()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        return this1

    @staticmethod
    def equals(this1,b):
        return (not (((((Reflect.field(Math,"fabs")((this1.x - b.x)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1.y - b.y)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1.z - b.z)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1.w - b.w)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return (((((((("{" + Std.string(this1.x)) + ", ") + Std.string(this1.y)) + ", ") + Std.string(this1.z)) + ", ") + Std.string(this1.w)) + "}")

    @staticmethod
    def lengthSquared(this1):
        return ((((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z))) + ((this1.w * this1.w)))

    @staticmethod
    def length(this1):
        v = ((((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z))) + ((this1.w * this1.w)))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def normalize(q,dest):
        v = ((((q.x * q.x) + ((q.y * q.y))) + ((q.z * q.z))) + ((q.w * q.w)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (q.x * mult)
        dest.y = (q.y * mult)
        dest.z = (q.z * mult)
        dest.w = (q.w * mult)
        return dest

    @staticmethod
    def dot(a,b):
        return ((((a.x * b.x) + ((a.y * b.y))) + ((a.z * b.z))) + ((a.w * b.w)))

    @staticmethod
    def identity(dest):
        dest.x = 0
        dest.y = 0
        dest.z = 0
        dest.w = 1
        return dest

    @staticmethod
    def copy(src,dest):
        dest.x = src.x
        dest.y = src.y
        dest.z = src.z
        dest.w = src.w
        return dest

    @staticmethod
    def axisAngle(axis,angle,dest):
        angle = (angle * 0.5)
        s = (Math.NaN if (((angle == Math.POSITIVE_INFINITY) or ((angle == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(angle))
        dest.x = (s * axis.x)
        dest.y = (s * axis.y)
        dest.z = (s * axis.z)
        dest.w = (Math.NaN if (((angle == Math.POSITIVE_INFINITY) or ((angle == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(angle))
        return dest

    @staticmethod
    def multiplyQuats(a,b,dest):
        ax = a.x
        ay = a.y
        az = a.z
        aw = a.w
        bx = b.x
        by = b.y
        bz = b.z
        bw = b.w
        dest.x = ((((ax * bw) + ((aw * bx))) + ((ay * bz))) - ((az * by)))
        dest.y = ((((ay * bw) + ((aw * by))) + ((az * bx))) - ((ax * bz)))
        dest.z = ((((az * bw) + ((aw * bz))) + ((ax * by))) - ((ay * bx)))
        dest.w = ((((aw * bw) - ((ax * bx))) - ((ay * by))) - ((az * bz)))
        return dest

    @staticmethod
    def multiplyQuatsOp(a,b):
        this1 = glm_QuatBase()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 1
        dest = this1
        ax = a.x
        ay = a.y
        az = a.z
        aw = a.w
        bx = b.x
        by = b.y
        bz = b.z
        bw = b.w
        dest.x = ((((ax * bw) + ((aw * bx))) + ((ay * bz))) - ((az * by)))
        dest.y = ((((ay * bw) + ((aw * by))) + ((az * bx))) - ((ax * bz)))
        dest.z = ((((az * bw) + ((aw * bz))) + ((ax * by))) - ((ay * bx)))
        dest.w = ((((aw * bw) - ((ax * bx))) - ((ay * by))) - ((az * bz)))
        return dest

    @staticmethod
    def multiplyScalar(a,s,dest):
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        dest.w = (a.w * s)
        return dest

    @staticmethod
    def multiplyScalarOp(a,s):
        this1 = glm_QuatBase()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 1
        dest = this1
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        dest.w = (a.w * s)
        return dest

    @staticmethod
    def lerp(a,b,t,dest):
        a1 = a.x
        dest.x = (a1 + ((t * ((b.x - a1)))))
        a1 = a.y
        dest.y = (a1 + ((t * ((b.y - a1)))))
        a1 = a.z
        dest.z = (a1 + ((t * ((b.z - a1)))))
        a1 = a.w
        dest.w = (a1 + ((t * ((b.w - a1)))))
        return dest

    @staticmethod
    def slerp(a,b,t,dest):
        bx = b.x
        by = b.y
        bz = b.z
        bw = b.w
        cosTheta = ((((a.x * b.x) + ((a.y * b.y))) + ((a.z * b.z))) + ((a.w * b.w)))
        if (cosTheta < 0):
            cosTheta = -cosTheta
            bx = -bx
            by = -by
            bz = -bz
            bw = -bw
        if (cosTheta > ((1 - glm_GLM.EPSILON))):
            a1 = a.x
            dest.x = (a1 + ((t * ((b.x - a1)))))
            a1 = a.y
            dest.y = (a1 + ((t * ((b.y - a1)))))
            a1 = a.z
            dest.z = (a1 + ((t * ((b.z - a1)))))
            a1 = a.w
            dest.w = (a1 + ((t * ((b.w - a1)))))
            return dest
        else:
            angle = Math.acos(cosTheta)
            sa = (1 / ((Math.NaN if (((angle == Math.POSITIVE_INFINITY) or ((angle == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(angle))))
            v = (((1 - t)) * angle)
            i = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
            v = (t * angle)
            j = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
            dest.x = ((((i * a.x) + ((j * bx)))) * sa)
            dest.y = ((((i * a.y) + ((j * by)))) * sa)
            dest.z = ((((i * a.z) + ((j * bz)))) * sa)
            dest.w = ((((i * a.w) + ((j * bw)))) * sa)
            return dest

    @staticmethod
    def invert(q,dest):
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        d = ((((q.x * q.x) + ((q.y * q.y))) + ((q.z * q.z))) + ((q.w * q.w)))
        oneOverD = (0 if ((d < glm_GLM.EPSILON)) else (1 / d))
        dest.x = (-x * oneOverD)
        dest.y = (-y * oneOverD)
        dest.z = (-z * oneOverD)
        dest.w = (w * oneOverD)
        return dest

    @staticmethod
    def conjugate(q,dest):
        dest.x = (-1 * q.x)
        dest.y = (-1 * q.y)
        dest.z = (-1 * q.z)
        dest.w = q.w
        return dest

    @staticmethod
    def fromEuler(x,y,z,dest):
        v = (x / 2)
        c1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
        v = (y / 2)
        c2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
        v = (z / 2)
        c3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(v))
        v = (x / 2)
        s1 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
        v = (y / 2)
        s2 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
        v = (z / 2)
        s3 = (Math.NaN if (((v == Math.POSITIVE_INFINITY) or ((v == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(v))
        dest.x = (((s1 * c2) * c3) + (((c1 * s2) * s3)))
        dest.y = (((c1 * s2) * c3) - (((s1 * c2) * s3)))
        dest.z = (((c1 * c2) * s3) + (((s1 * s2) * c3)))
        dest.w = (((c1 * c2) * c3) - (((s1 * s2) * s3)))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        x = (arr[0] if 0 < len(arr) else None)
        y = (arr[1] if 1 < len(arr) else None)
        z = (arr[2] if 2 < len(arr) else None)
        w = (arr[3] if 3 < len(arr) else None)
        if (w is None):
            w = 1
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_QuatBase()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1.x, this1.y, this1.z, this1.w]
glm__Quat_Quat_Impl_._hx_class = glm__Quat_Quat_Impl_


class glm_Vec2Base:
    _hx_class_name = "glm.Vec2Base"
    __slots__ = ("x", "y")
    _hx_fields = ["x", "y"]

    def __init__(self):
        self.y = None
        self.x = None

glm_Vec2Base._hx_class = glm_Vec2Base


class glm__Vec2_Vec2_Impl_:
    _hx_class_name = "glm._Vec2.Vec2_Impl_"
    __slots__ = ()
    _hx_statics = ["get_x", "set_x", "get_y", "set_y", "get_i", "set_i", "get_j", "set_j", "get", "arrayWrite", "_new", "equals", "toString", "lengthSquared", "length", "copy", "set", "addVec", "subtractVec", "addVecOp", "subtractVecOp", "addScalar", "multiplyScalar", "addScalarOp", "subtractScalarOp", "multiplyScalarOp", "divideScalarOp", "distanceSquared", "distance", "dot", "cross", "normalize", "lerp", "fromFloatArray", "toFloatArray"]
    x = None
    y = None
    i = None
    j = None

    @staticmethod
    def get_x(this1):
        return this1.x

    @staticmethod
    def set_x(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_y(this1):
        return this1.y

    @staticmethod
    def set_y(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_i(this1):
        return this1.x

    @staticmethod
    def set_i(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_j(this1):
        return this1.y

    @staticmethod
    def set_j(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1.x
        elif (key1 == 1):
            return this1.y
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-1)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1.x = value
                    return this1.x
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1.y = value
                    return this1.y
                return _hx_local_2()
            return _hx_local_3()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-1)!"))

    @staticmethod
    def _new(x = None,y = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    @staticmethod
    def equals(this1,b):
        return (not (((Reflect.field(Math,"fabs")((this1.x - b.x)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1.y - b.y)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return (((("<" + Std.string(this1.x)) + ", ") + Std.string(this1.y)) + ">")

    @staticmethod
    def lengthSquared(this1):
        return ((this1.x * this1.x) + ((this1.y * this1.y)))

    @staticmethod
    def length(this1):
        v = ((this1.x * this1.x) + ((this1.y * this1.y)))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def copy(src,dest):
        dest.x = src.x
        dest.y = src.y
        return dest

    @staticmethod
    def set(dest,x = None,y = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        dest.x = x
        dest.y = y
        return dest

    @staticmethod
    def addVec(a,b,dest):
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        return dest

    @staticmethod
    def subtractVec(a,b,dest):
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        return dest

    @staticmethod
    def addVecOp(a,b):
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        return dest

    @staticmethod
    def subtractVecOp(a,b):
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        return dest

    @staticmethod
    def addScalar(a,s,dest):
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        return dest

    @staticmethod
    def multiplyScalar(a,s,dest):
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        return dest

    @staticmethod
    def addScalarOp(a,s):
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        return dest

    @staticmethod
    def subtractScalarOp(a,s):
        s1 = -s
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x + s1)
        dest.y = (a.y + s1)
        return dest

    @staticmethod
    def multiplyScalarOp(a,s):
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        return dest

    @staticmethod
    def divideScalarOp(a,s):
        s1 = (1 / s)
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (a.x * s1)
        dest.y = (a.y * s1)
        return dest

    @staticmethod
    def distanceSquared(a,b):
        return ((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y)))))

    @staticmethod
    def distance(a,b):
        v = ((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y)))))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def dot(a,b):
        return ((a.x * b.x) + ((a.y * b.y)))

    @staticmethod
    def cross(a,b,dest):
        x = 0
        y = 0
        z = ((a.x * b.y) - ((a.y * b.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest = this1
        return dest

    @staticmethod
    def normalize(v,dest):
        v1 = ((v.x * v.x) + ((v.y * v.y)))
        length = (Math.NaN if ((v1 < 0)) else python_lib_Math.sqrt(v1))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (v.x * mult)
        dest.y = (v.y * mult)
        return dest

    @staticmethod
    def lerp(a,b,t,dest):
        a1 = a.x
        dest.x = (a1 + ((t * ((b.x - a1)))))
        a1 = a.y
        dest.y = (a1 + ((t * ((b.y - a1)))))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        x = (arr[0] if 0 < len(arr) else None)
        y = (arr[1] if 1 < len(arr) else None)
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1.x, this1.y]
glm__Vec2_Vec2_Impl_._hx_class = glm__Vec2_Vec2_Impl_


class glm_Vec3Base:
    _hx_class_name = "glm.Vec3Base"
    __slots__ = ("x", "y", "z")
    _hx_fields = ["x", "y", "z"]

    def __init__(self):
        self.z = None
        self.y = None
        self.x = None

glm_Vec3Base._hx_class = glm_Vec3Base


class glm__Vec3_Vec3_Impl_:
    _hx_class_name = "glm._Vec3.Vec3_Impl_"
    __slots__ = ()
    _hx_statics = ["get_x", "set_x", "get_y", "set_y", "get_z", "set_z", "get_r", "set_r", "get_g", "set_g", "get_b", "set_b", "get", "arrayWrite", "_new", "equals", "toString", "lengthSquared", "length", "copy", "set", "addVec", "subtractVec", "addVecOp", "subtractVecOp", "addScalar", "multiplyScalar", "addScalarOp", "subtractScalarOp", "multiplyScalarOp", "divideScalarOp", "distanceSquared", "distance", "dot", "cross", "normalize", "lerp", "fromFloatArray", "toFloatArray"]
    x = None
    y = None
    z = None
    r = None
    g = None
    b = None

    @staticmethod
    def get_x(this1):
        return this1.x

    @staticmethod
    def set_x(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_y(this1):
        return this1.y

    @staticmethod
    def set_y(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_z(this1):
        return this1.z

    @staticmethod
    def set_z(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.z = v
                return this1.z
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r(this1):
        return this1.x

    @staticmethod
    def set_r(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_g(this1):
        return this1.y

    @staticmethod
    def set_g(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_b(this1):
        return this1.z

    @staticmethod
    def set_b(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.z = v
                return this1.z
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1.x
        elif (key1 == 1):
            return this1.y
        elif (key1 == 2):
            return this1.z
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-2)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1.x = value
                    return this1.x
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1.y = value
                    return this1.y
                return _hx_local_2()
            return _hx_local_3()
        elif (key1 == 2):
            def _hx_local_5():
                def _hx_local_4():
                    this1.z = value
                    return this1.z
                return _hx_local_4()
            return _hx_local_5()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-2)!"))

    @staticmethod
    def _new(x = None,y = None,z = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        if (z is None):
            z = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        return this1

    @staticmethod
    def equals(this1,b):
        return (not ((((Reflect.field(Math,"fabs")((this1.x - b.x)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1.y - b.y)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1.z - b.z)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return (((((("<" + Std.string(this1.x)) + ", ") + Std.string(this1.y)) + ", ") + Std.string(this1.z)) + ">")

    @staticmethod
    def lengthSquared(this1):
        return (((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z)))

    @staticmethod
    def length(this1):
        v = (((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z)))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def copy(src,dest):
        dest.x = src.x
        dest.y = src.y
        dest.z = src.z
        return dest

    @staticmethod
    def set(dest,x = None,y = None,z = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        if (z is None):
            z = 0
        dest.x = x
        dest.y = y
        dest.z = z
        return dest

    @staticmethod
    def addVec(a,b,dest):
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        dest.z = (a.z + b.z)
        return dest

    @staticmethod
    def subtractVec(a,b,dest):
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        dest.z = (a.z - b.z)
        return dest

    @staticmethod
    def addVecOp(a,b):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        dest.z = (a.z + b.z)
        return dest

    @staticmethod
    def subtractVecOp(a,b):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        dest.z = (a.z - b.z)
        return dest

    @staticmethod
    def addScalar(a,s,dest):
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        dest.z = (a.z + s)
        return dest

    @staticmethod
    def multiplyScalar(a,s,dest):
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        return dest

    @staticmethod
    def addScalarOp(a,s):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        dest.z = (a.z + s)
        return dest

    @staticmethod
    def subtractScalarOp(a,s):
        s1 = -s
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x + s1)
        dest.y = (a.y + s1)
        dest.z = (a.z + s1)
        return dest

    @staticmethod
    def multiplyScalarOp(a,s):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        return dest

    @staticmethod
    def divideScalarOp(a,s):
        s1 = (1 / s)
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (a.x * s1)
        dest.y = (a.y * s1)
        dest.z = (a.z * s1)
        return dest

    @staticmethod
    def distanceSquared(a,b):
        return (((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y))))) + ((((a.z - b.z)) * ((a.z - b.z)))))

    @staticmethod
    def distance(a,b):
        v = (((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y))))) + ((((a.z - b.z)) * ((a.z - b.z)))))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def dot(a,b):
        return (((a.x * b.x) + ((a.y * b.y))) + ((a.z * b.z)))

    @staticmethod
    def cross(a,b,dest):
        x = ((a.y * b.z) - ((a.z * b.y)))
        y = ((a.z * b.x) - ((a.x * b.z)))
        z = ((a.x * b.y) - ((a.y * b.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest = this1
        return dest

    @staticmethod
    def normalize(v,dest):
        v1 = (((v.x * v.x) + ((v.y * v.y))) + ((v.z * v.z)))
        length = (Math.NaN if ((v1 < 0)) else python_lib_Math.sqrt(v1))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (v.x * mult)
        dest.y = (v.y * mult)
        dest.z = (v.z * mult)
        return dest

    @staticmethod
    def lerp(a,b,t,dest):
        a1 = a.x
        dest.x = (a1 + ((t * ((b.x - a1)))))
        a1 = a.y
        dest.y = (a1 + ((t * ((b.y - a1)))))
        a1 = a.z
        dest.z = (a1 + ((t * ((b.z - a1)))))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        x = (arr[0] if 0 < len(arr) else None)
        y = (arr[1] if 1 < len(arr) else None)
        z = (arr[2] if 2 < len(arr) else None)
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1.x, this1.y, this1.z]
glm__Vec3_Vec3_Impl_._hx_class = glm__Vec3_Vec3_Impl_


class glm_Vec4Base:
    _hx_class_name = "glm.Vec4Base"
    __slots__ = ("x", "y", "z", "w")
    _hx_fields = ["x", "y", "z", "w"]

    def __init__(self):
        self.w = None
        self.z = None
        self.y = None
        self.x = None

glm_Vec4Base._hx_class = glm_Vec4Base


class glm__Vec4_Vec4_Impl_:
    _hx_class_name = "glm._Vec4.Vec4_Impl_"
    __slots__ = ()
    _hx_statics = ["get_x", "set_x", "get_y", "set_y", "get_z", "set_z", "get_w", "set_w", "get_r", "set_r", "get_g", "set_g", "get_b", "set_b", "get_a", "set_a", "get", "arrayWrite", "_new", "equals", "toString", "lengthSquared", "length", "copy", "set", "addVec", "subtractVec", "addVecOp", "subtractVecOp", "addScalar", "multiplyScalar", "addScalarOp", "subtractScalarOp", "multiplyScalarOp", "divideScalarOp", "distanceSquared", "distance", "dot", "normalize", "lerp", "fromFloatArray", "toFloatArray"]
    x = None
    y = None
    z = None
    w = None
    r = None
    g = None
    b = None
    a = None

    @staticmethod
    def get_x(this1):
        return this1.x

    @staticmethod
    def set_x(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_y(this1):
        return this1.y

    @staticmethod
    def set_y(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_z(this1):
        return this1.z

    @staticmethod
    def set_z(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.z = v
                return this1.z
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_w(this1):
        return this1.w

    @staticmethod
    def set_w(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.w = v
                return this1.w
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_r(this1):
        return this1.x

    @staticmethod
    def set_r(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.x = v
                return this1.x
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_g(this1):
        return this1.y

    @staticmethod
    def set_g(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.y = v
                return this1.y
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_b(this1):
        return this1.z

    @staticmethod
    def set_b(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.z = v
                return this1.z
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get_a(this1):
        return this1.w

    @staticmethod
    def set_a(this1,v):
        def _hx_local_1():
            def _hx_local_0():
                this1.w = v
                return this1.w
            return _hx_local_0()
        return _hx_local_1()

    @staticmethod
    def get(this1,key):
        key1 = key
        if (key1 == 0):
            return this1.x
        elif (key1 == 1):
            return this1.y
        elif (key1 == 2):
            return this1.z
        elif (key1 == 3):
            return this1.w
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-3)!"))

    @staticmethod
    def arrayWrite(this1,key,value):
        key1 = key
        if (key1 == 0):
            def _hx_local_1():
                def _hx_local_0():
                    this1.x = value
                    return this1.x
                return _hx_local_0()
            return _hx_local_1()
        elif (key1 == 1):
            def _hx_local_3():
                def _hx_local_2():
                    this1.y = value
                    return this1.y
                return _hx_local_2()
            return _hx_local_3()
        elif (key1 == 2):
            def _hx_local_5():
                def _hx_local_4():
                    this1.z = value
                    return this1.z
                return _hx_local_4()
            return _hx_local_5()
        elif (key1 == 3):
            def _hx_local_7():
                def _hx_local_6():
                    this1.w = value
                    return this1.w
                return _hx_local_6()
            return _hx_local_7()
        else:
            raise haxe_Exception.thrown((("Index " + Std.string(key)) + " out of bounds (0-3)!"))

    @staticmethod
    def _new(x = None,y = None,z = None,w = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        if (z is None):
            z = 0
        if (w is None):
            w = 0
        this1 = glm_Vec4Base()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        return this1

    @staticmethod
    def equals(this1,b):
        return (not (((((Reflect.field(Math,"fabs")((this1.x - b.x)) >= glm_GLM.EPSILON) or ((Reflect.field(Math,"fabs")((this1.y - b.y)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1.z - b.z)) >= glm_GLM.EPSILON))) or ((Reflect.field(Math,"fabs")((this1.w - b.w)) >= glm_GLM.EPSILON)))))

    @staticmethod
    def toString(this1):
        return (((((((("<" + Std.string(this1.x)) + ", ") + Std.string(this1.y)) + ", ") + Std.string(this1.z)) + ", ") + Std.string(this1.w)) + ">")

    @staticmethod
    def lengthSquared(this1):
        return ((((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z))) + ((this1.w * this1.w)))

    @staticmethod
    def length(this1):
        v = ((((this1.x * this1.x) + ((this1.y * this1.y))) + ((this1.z * this1.z))) + ((this1.w * this1.w)))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def copy(src,dest):
        dest.x = src.x
        dest.y = src.y
        dest.z = src.z
        dest.w = src.w
        return dest

    @staticmethod
    def set(dest,x = None,y = None,z = None,w = None):
        if (x is None):
            x = 0
        if (y is None):
            y = 0
        if (z is None):
            z = 0
        if (w is None):
            w = 0
        dest.x = x
        dest.y = y
        dest.z = z
        dest.w = w
        return dest

    @staticmethod
    def addVec(a,b,dest):
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        dest.z = (a.z + b.z)
        dest.w = (a.w + b.w)
        return dest

    @staticmethod
    def subtractVec(a,b,dest):
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        dest.z = (a.z - b.z)
        dest.w = (a.w - b.w)
        return dest

    @staticmethod
    def addVecOp(a,b):
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x + b.x)
        dest.y = (a.y + b.y)
        dest.z = (a.z + b.z)
        dest.w = (a.w + b.w)
        return dest

    @staticmethod
    def subtractVecOp(a,b):
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x - b.x)
        dest.y = (a.y - b.y)
        dest.z = (a.z - b.z)
        dest.w = (a.w - b.w)
        return dest

    @staticmethod
    def addScalar(a,s,dest):
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        dest.z = (a.z + s)
        dest.w = (a.w + s)
        return dest

    @staticmethod
    def multiplyScalar(a,s,dest):
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        dest.w = (a.w * s)
        return dest

    @staticmethod
    def addScalarOp(a,s):
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x + s)
        dest.y = (a.y + s)
        dest.z = (a.z + s)
        dest.w = (a.w + s)
        return dest

    @staticmethod
    def subtractScalarOp(a,s):
        s1 = -s
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x + s1)
        dest.y = (a.y + s1)
        dest.z = (a.z + s1)
        dest.w = (a.w + s1)
        return dest

    @staticmethod
    def multiplyScalarOp(a,s):
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x * s)
        dest.y = (a.y * s)
        dest.z = (a.z * s)
        dest.w = (a.w * s)
        return dest

    @staticmethod
    def divideScalarOp(a,s):
        s1 = (1 / s)
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        dest = this1
        dest.x = (a.x * s1)
        dest.y = (a.y * s1)
        dest.z = (a.z * s1)
        dest.w = (a.w * s1)
        return dest

    @staticmethod
    def distanceSquared(a,b):
        return ((((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y))))) + ((((a.z - b.z)) * ((a.z - b.z))))) + ((((a.w - b.w)) * ((a.w - b.w)))))

    @staticmethod
    def distance(a,b):
        v = ((((((a.x - b.x)) * ((a.x - b.x))) + ((((a.y - b.y)) * ((a.y - b.y))))) + ((((a.z - b.z)) * ((a.z - b.z))))) + ((((a.w - b.w)) * ((a.w - b.w)))))
        if (v < 0):
            return Math.NaN
        else:
            return python_lib_Math.sqrt(v)

    @staticmethod
    def dot(a,b):
        return ((((a.x * b.x) + ((a.y * b.y))) + ((a.z * b.z))) + ((a.w * b.w)))

    @staticmethod
    def normalize(v,dest):
        v1 = ((((v.x * v.x) + ((v.y * v.y))) + ((v.z * v.z))) + ((v.w * v.w)))
        length = (Math.NaN if ((v1 < 0)) else python_lib_Math.sqrt(v1))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (v.x * mult)
        dest.y = (v.y * mult)
        dest.z = (v.z * mult)
        dest.w = (v.w * mult)
        return dest

    @staticmethod
    def lerp(a,b,t,dest):
        a1 = a.x
        dest.x = (a1 + ((t * ((b.x - a1)))))
        a1 = a.y
        dest.y = (a1 + ((t * ((b.y - a1)))))
        a1 = a.z
        dest.z = (a1 + ((t * ((b.z - a1)))))
        a1 = a.w
        dest.w = (a1 + ((t * ((b.w - a1)))))
        return dest

    @staticmethod
    def fromFloatArray(arr):
        x = (arr[0] if 0 < len(arr) else None)
        y = (arr[1] if 1 < len(arr) else None)
        z = (arr[2] if 2 < len(arr) else None)
        w = (arr[3] if 3 < len(arr) else None)
        if (w is None):
            w = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec4Base()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        return this1

    @staticmethod
    def toFloatArray(this1):
        return [this1.x, this1.y, this1.z, this1.w]
glm__Vec4_Vec4_Impl_._hx_class = glm__Vec4_Vec4_Impl_

class haxe_StackItem(Enum):
    __slots__ = ()
    _hx_class_name = "haxe.StackItem"
    _hx_constructs = ["CFunction", "Module", "FilePos", "Method", "LocalFunction"]

    @staticmethod
    def Module(m):
        return haxe_StackItem("Module", 1, (m,))

    @staticmethod
    def FilePos(s,file,line,column = None):
        return haxe_StackItem("FilePos", 2, (s,file,line,column))

    @staticmethod
    def Method(classname,method):
        return haxe_StackItem("Method", 3, (classname,method))

    @staticmethod
    def LocalFunction(v = None):
        return haxe_StackItem("LocalFunction", 4, (v,))
haxe_StackItem.CFunction = haxe_StackItem("CFunction", 0, ())
haxe_StackItem._hx_class = haxe_StackItem


class haxe__CallStack_CallStack_Impl_:
    _hx_class_name = "haxe._CallStack.CallStack_Impl_"
    __slots__ = ()
    _hx_statics = ["callStack", "exceptionStack", "subtract", "equalItems"]

    @staticmethod
    def callStack():
        infos = python_lib_Traceback.extract_stack()
        if (len(infos) != 0):
            infos.pop()
        infos.reverse()
        return haxe_NativeStackTrace.toHaxe(infos)

    @staticmethod
    def exceptionStack(fullStack = None):
        if (fullStack is None):
            fullStack = False
        eStack = haxe_NativeStackTrace.toHaxe(haxe_NativeStackTrace.exceptionStack())
        return (eStack if fullStack else haxe__CallStack_CallStack_Impl_.subtract(eStack,haxe__CallStack_CallStack_Impl_.callStack()))

    @staticmethod
    def subtract(this1,stack):
        startIndex = -1
        i = -1
        while True:
            i = (i + 1)
            tmp = i
            if (not ((tmp < len(this1)))):
                break
            _g = 0
            _g1 = len(stack)
            while (_g < _g1):
                j = _g
                _g = (_g + 1)
                if haxe__CallStack_CallStack_Impl_.equalItems((this1[i] if i >= 0 and i < len(this1) else None),python_internal_ArrayImpl._get(stack, j)):
                    if (startIndex < 0):
                        startIndex = i
                    i = (i + 1)
                    if (i >= len(this1)):
                        break
                else:
                    startIndex = -1
            if (startIndex >= 0):
                break
        if (startIndex >= 0):
            return this1[0:startIndex]
        else:
            return this1

    @staticmethod
    def equalItems(item1,item2):
        if (item1 is None):
            if (item2 is None):
                return True
            else:
                return False
        else:
            tmp = item1.index
            if (tmp == 0):
                if (item2 is None):
                    return False
                elif (item2.index == 0):
                    return True
                else:
                    return False
            elif (tmp == 1):
                if (item2 is None):
                    return False
                elif (item2.index == 1):
                    m2 = item2.params[0]
                    m1 = item1.params[0]
                    return (m1 == m2)
                else:
                    return False
            elif (tmp == 2):
                if (item2 is None):
                    return False
                elif (item2.index == 2):
                    item21 = item2.params[0]
                    file2 = item2.params[1]
                    line2 = item2.params[2]
                    col2 = item2.params[3]
                    col1 = item1.params[3]
                    line1 = item1.params[2]
                    file1 = item1.params[1]
                    item11 = item1.params[0]
                    if (((file1 == file2) and ((line1 == line2))) and ((col1 == col2))):
                        return haxe__CallStack_CallStack_Impl_.equalItems(item11,item21)
                    else:
                        return False
                else:
                    return False
            elif (tmp == 3):
                if (item2 is None):
                    return False
                elif (item2.index == 3):
                    class2 = item2.params[0]
                    method2 = item2.params[1]
                    method1 = item1.params[1]
                    class1 = item1.params[0]
                    if (class1 == class2):
                        return (method1 == method2)
                    else:
                        return False
                else:
                    return False
            elif (tmp == 4):
                if (item2 is None):
                    return False
                elif (item2.index == 4):
                    v2 = item2.params[0]
                    v1 = item1.params[0]
                    return (v1 == v2)
                else:
                    return False
            else:
                pass
haxe__CallStack_CallStack_Impl_._hx_class = haxe__CallStack_CallStack_Impl_


class haxe_IMap:
    _hx_class_name = "haxe.IMap"
    __slots__ = ()
haxe_IMap._hx_class = haxe_IMap


class haxe_Exception(Exception):
    _hx_class_name = "haxe.Exception"
    __slots__ = ("_hx___nativeStack", "_hx___skipStack", "_hx___nativeException", "_hx___previousException")
    _hx_fields = ["__nativeStack", "__skipStack", "__nativeException", "__previousException"]
    _hx_methods = ["unwrap", "get_native"]
    _hx_statics = ["caught", "thrown"]
    _hx_interfaces = []
    _hx_super = Exception


    def __init__(self,message,previous = None,native = None):
        self._hx___previousException = None
        self._hx___nativeException = None
        self._hx___nativeStack = None
        self._hx___skipStack = 0
        super().__init__(message)
        self._hx___previousException = previous
        if ((native is not None) and Std.isOfType(native,BaseException)):
            self._hx___nativeException = native
            self._hx___nativeStack = haxe_NativeStackTrace.exceptionStack()
        else:
            self._hx___nativeException = self
            infos = python_lib_Traceback.extract_stack()
            if (len(infos) != 0):
                infos.pop()
            infos.reverse()
            self._hx___nativeStack = infos

    def unwrap(self):
        return self._hx___nativeException

    def get_native(self):
        return self._hx___nativeException

    @staticmethod
    def caught(value):
        if Std.isOfType(value,haxe_Exception):
            return value
        elif Std.isOfType(value,BaseException):
            return haxe_Exception(str(value),None,value)
        else:
            return haxe_ValueException(value,None,value)

    @staticmethod
    def thrown(value):
        if Std.isOfType(value,haxe_Exception):
            return value.get_native()
        elif Std.isOfType(value,BaseException):
            return value
        else:
            e = haxe_ValueException(value)
            e._hx___skipStack = (e._hx___skipStack + 1)
            return e

haxe_Exception._hx_class = haxe_Exception


class haxe__Int32_Int32_Impl_:
    _hx_class_name = "haxe._Int32.Int32_Impl_"
    __slots__ = ()
    _hx_statics = ["ucompare"]

    @staticmethod
    def ucompare(a,b):
        if (a < 0):
            if (b < 0):
                return (((((~b + (2 ** 31)) % (2 ** 32) - (2 ** 31)) - (((~a + (2 ** 31)) % (2 ** 32) - (2 ** 31)))) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
            else:
                return 1
        if (b < 0):
            return -1
        else:
            return (((a - b) + (2 ** 31)) % (2 ** 32) - (2 ** 31))
haxe__Int32_Int32_Impl_._hx_class = haxe__Int32_Int32_Impl_


class haxe__Int64____Int64:
    _hx_class_name = "haxe._Int64.___Int64"
    __slots__ = ("high", "low")
    _hx_fields = ["high", "low"]

    def __init__(self,high,low):
        self.high = high
        self.low = low

haxe__Int64____Int64._hx_class = haxe__Int64____Int64


class haxe_Log:
    _hx_class_name = "haxe.Log"
    __slots__ = ()
    _hx_statics = ["formatOutput", "trace"]

    @staticmethod
    def formatOutput(v,infos):
        _hx_str = Std.string(v)
        if (infos is None):
            return _hx_str
        pstr = ((HxOverrides.stringOrNull(infos.fileName) + ":") + Std.string(infos.lineNumber))
        if (Reflect.field(infos,"customParams") is not None):
            _g = 0
            _g1 = Reflect.field(infos,"customParams")
            while (_g < len(_g1)):
                v = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                _g = (_g + 1)
                _hx_str = (("null" if _hx_str is None else _hx_str) + ((", " + Std.string(v))))
        return ((("null" if pstr is None else pstr) + ": ") + ("null" if _hx_str is None else _hx_str))

    @staticmethod
    def trace(v,infos = None):
        _hx_str = haxe_Log.formatOutput(v,infos)
        str1 = Std.string(_hx_str)
        python_Lib.printString((("" + ("null" if str1 is None else str1)) + HxOverrides.stringOrNull(python_Lib.lineEnd)))
haxe_Log._hx_class = haxe_Log


class haxe_NativeStackTrace:
    _hx_class_name = "haxe.NativeStackTrace"
    __slots__ = ()
    _hx_statics = ["saveStack", "exceptionStack", "toHaxe"]

    @staticmethod
    def saveStack(exception):
        pass

    @staticmethod
    def exceptionStack():
        exc = python_lib_Sys.exc_info()
        if (exc[2] is not None):
            infos = python_lib_Traceback.extract_tb(exc[2])
            infos.reverse()
            return infos
        else:
            return []

    @staticmethod
    def toHaxe(native,skip = None):
        if (skip is None):
            skip = 0
        stack = []
        _g = 0
        _g1 = len(native)
        while (_g < _g1):
            i = _g
            _g = (_g + 1)
            if (skip > i):
                continue
            elem = (native[i] if i >= 0 and i < len(native) else None)
            x = haxe_StackItem.FilePos(haxe_StackItem.Method(None,elem[2]),elem[0],elem[1])
            stack.append(x)
        return stack
haxe_NativeStackTrace._hx_class = haxe_NativeStackTrace


class haxe_ValueException(haxe_Exception):
    _hx_class_name = "haxe.ValueException"
    __slots__ = ("value",)
    _hx_fields = ["value"]
    _hx_methods = ["unwrap"]
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = haxe_Exception


    def __init__(self,value,previous = None,native = None):
        self.value = None
        super().__init__(Std.string(value),previous,native)
        self.value = value

    def unwrap(self):
        return self.value

haxe_ValueException._hx_class = haxe_ValueException


class haxe_ds_IntMap:
    _hx_class_name = "haxe.ds.IntMap"
    __slots__ = ("h",)
    _hx_fields = ["h"]
    _hx_methods = ["set"]
    _hx_interfaces = [haxe_IMap]

    def __init__(self):
        self.h = dict()

    def set(self,key,value):
        self.h[key] = value

haxe_ds_IntMap._hx_class = haxe_ds_IntMap


class haxe_ds_List:
    _hx_class_name = "haxe.ds.List"
    __slots__ = ("h", "q", "length")
    _hx_fields = ["h", "q", "length"]
    _hx_methods = ["add", "pop", "isEmpty", "iterator", "filter"]

    def __init__(self):
        self.q = None
        self.h = None
        self.length = 0

    def add(self,item):
        x = haxe_ds__List_ListNode(item,None)
        if (self.h is None):
            self.h = x
        else:
            self.q.next = x
        self.q = x
        _hx_local_0 = self
        _hx_local_1 = _hx_local_0.length
        _hx_local_0.length = (_hx_local_1 + 1)
        _hx_local_1

    def pop(self):
        if (self.h is None):
            return None
        x = self.h.item
        self.h = self.h.next
        if (self.h is None):
            self.q = None
        _hx_local_0 = self
        _hx_local_1 = _hx_local_0.length
        _hx_local_0.length = (_hx_local_1 - 1)
        _hx_local_1
        return x

    def isEmpty(self):
        return (self.h is None)

    def iterator(self):
        return haxe_ds__List_ListIterator(self.h)

    def filter(self,f):
        l2 = haxe_ds_List()
        l = self.h
        while (l is not None):
            v = l.item
            l = l.next
            if f(v):
                l2.add(v)
        return l2

haxe_ds_List._hx_class = haxe_ds_List


class haxe_ds__List_ListNode:
    _hx_class_name = "haxe.ds._List.ListNode"
    __slots__ = ("item", "next")
    _hx_fields = ["item", "next"]

    def __init__(self,item,next):
        self.item = item
        self.next = next

haxe_ds__List_ListNode._hx_class = haxe_ds__List_ListNode


class haxe_ds__List_ListIterator:
    _hx_class_name = "haxe.ds._List.ListIterator"
    __slots__ = ("head",)
    _hx_fields = ["head"]
    _hx_methods = ["hasNext", "next"]

    def __init__(self,head):
        self.head = head

    def hasNext(self):
        return (self.head is not None)

    def next(self):
        val = self.head.item
        self.head = self.head.next
        return val

haxe_ds__List_ListIterator._hx_class = haxe_ds__List_ListIterator


class haxe_ds_ObjectMap:
    _hx_class_name = "haxe.ds.ObjectMap"
    __slots__ = ("h",)
    _hx_fields = ["h"]
    _hx_methods = ["set"]
    _hx_interfaces = [haxe_IMap]

    def __init__(self):
        self.h = dict()

    def set(self,key,value):
        self.h[key] = value

haxe_ds_ObjectMap._hx_class = haxe_ds_ObjectMap

class haxe_ds_Option(Enum):
    __slots__ = ()
    _hx_class_name = "haxe.ds.Option"
    _hx_constructs = ["Some", "None"]

    @staticmethod
    def Some(v):
        return haxe_ds_Option("Some", 0, (v,))
haxe_ds_Option._hx_None = haxe_ds_Option("None", 1, ())
haxe_ds_Option._hx_class = haxe_ds_Option


class haxe_iterators_ArrayIterator:
    _hx_class_name = "haxe.iterators.ArrayIterator"
    __slots__ = ("array", "current")
    _hx_fields = ["array", "current"]
    _hx_methods = ["hasNext", "next"]

    def __init__(self,array):
        self.current = 0
        self.array = array

    def hasNext(self):
        return (self.current < len(self.array))

    def next(self):
        def _hx_local_3():
            def _hx_local_2():
                _hx_local_0 = self
                _hx_local_1 = _hx_local_0.current
                _hx_local_0.current = (_hx_local_1 + 1)
                return _hx_local_1
            return python_internal_ArrayImpl._get(self.array, _hx_local_2())
        return _hx_local_3()

haxe_iterators_ArrayIterator._hx_class = haxe_iterators_ArrayIterator


class haxe_iterators_ArrayKeyValueIterator:
    _hx_class_name = "haxe.iterators.ArrayKeyValueIterator"
    __slots__ = ("current", "array")
    _hx_fields = ["current", "array"]
    _hx_methods = ["hasNext", "next"]

    def __init__(self,array):
        self.current = 0
        self.array = array

    def hasNext(self):
        return (self.current < len(self.array))

    def next(self):
        def _hx_local_3():
            def _hx_local_2():
                _hx_local_0 = self
                _hx_local_1 = _hx_local_0.current
                _hx_local_0.current = (_hx_local_1 + 1)
                return _hx_local_1
            return _hx_AnonObject({'value': python_internal_ArrayImpl._get(self.array, self.current), 'key': _hx_local_2()})
        return _hx_local_3()

haxe_iterators_ArrayKeyValueIterator._hx_class = haxe_iterators_ArrayKeyValueIterator


class haxe_rtti_Meta:
    _hx_class_name = "haxe.rtti.Meta"
    __slots__ = ()
    _hx_statics = ["getType", "getMeta"]

    @staticmethod
    def getType(t):
        meta = haxe_rtti_Meta.getMeta(t)
        if ((meta is None) or ((Reflect.field(meta,"obj") is None))):
            return _hx_AnonObject({})
        else:
            return Reflect.field(meta,"obj")

    @staticmethod
    def getMeta(t):
        return Reflect.field(t,"__meta__")
haxe_rtti_Meta._hx_class = haxe_rtti_Meta

class headbutt_threed__Headbutt_EvolveResult(Enum):
    __slots__ = ()
    _hx_class_name = "headbutt.threed._Headbutt.EvolveResult"
    _hx_constructs = ["NoIntersection", "FoundIntersection", "StillEvolving"]
headbutt_threed__Headbutt_EvolveResult.NoIntersection = headbutt_threed__Headbutt_EvolveResult("NoIntersection", 0, ())
headbutt_threed__Headbutt_EvolveResult.FoundIntersection = headbutt_threed__Headbutt_EvolveResult("FoundIntersection", 1, ())
headbutt_threed__Headbutt_EvolveResult.StillEvolving = headbutt_threed__Headbutt_EvolveResult("StillEvolving", 2, ())
headbutt_threed__Headbutt_EvolveResult._hx_class = headbutt_threed__Headbutt_EvolveResult


class headbutt_threed_Headbutt:
    _hx_class_name = "headbutt.threed.Headbutt"
    __slots__ = ()
    _hx_statics = ["MAX_ITERATIONS", "calculateSupport", "addSupport", "evolveSimplex", "test"]

    def __init__(self):
        pass

    @staticmethod
    def calculateSupport(shapeA,shapeB,direction):
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = (direction.x * -1)
        dest.y = (direction.y * -1)
        dest.z = (direction.z * -1)
        oppositeDirection = dest
        src = shapeA.support(direction)
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = src.x
        dest.y = src.y
        dest.z = src.z
        newVertex = dest
        b = shapeB.support(oppositeDirection)
        _hx_local_0 = newVertex
        _hx_local_1 = _hx_local_0.x
        _hx_local_0.x = (_hx_local_1 - b.x)
        _hx_local_0.x
        _hx_local_2 = newVertex
        _hx_local_3 = _hx_local_2.y
        _hx_local_2.y = (_hx_local_3 - b.y)
        _hx_local_2.y
        _hx_local_4 = newVertex
        _hx_local_5 = _hx_local_4.z
        _hx_local_4.z = (_hx_local_5 - b.z)
        _hx_local_4.z
        return newVertex

    @staticmethod
    def addSupport(vertices,shapeA,shapeB,direction):
        newVertex = headbutt_threed_Headbutt.calculateSupport(shapeA,shapeB,direction)
        vertices.append(newVertex)
        return ((((direction.x * newVertex.x) + ((direction.y * newVertex.y))) + ((direction.z * newVertex.z))) >= 0)

    @staticmethod
    def evolveSimplex(vertices,shapeA,shapeB,direction):
        _g = len(vertices)
        if (_g == 0):
            a = shapeB.get_centre()
            b = shapeA.get_centre()
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            direction = dest
        elif (_g == 1):
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (direction.x * -1)
            dest.y = (direction.y * -1)
            dest.z = (direction.z * -1)
            direction = dest
        elif (_g == 2):
            a = (vertices[1] if 1 < len(vertices) else None)
            b = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            ab = dest
            a = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x * -1)
            dest.y = (a.y * -1)
            dest.z = (a.z * -1)
            a0 = dest
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            x = ((ab.y * a0.z) - ((ab.z * a0.y)))
            y = ((ab.z * a0.x) - ((ab.x * a0.z)))
            z = ((ab.x * a0.y) - ((ab.y * a0.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            tmp = dest
            dest = direction
            x = ((tmp.y * ab.z) - ((tmp.z * ab.y)))
            y = ((tmp.z * ab.x) - ((tmp.x * ab.z)))
            z = ((tmp.x * ab.y) - ((tmp.y * ab.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            direction = dest
        elif (_g == 3):
            a = (vertices[2] if 2 < len(vertices) else None)
            b = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            ac = dest
            a = (vertices[1] if 1 < len(vertices) else None)
            b = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            ab = dest
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            x = ((ac.y * ab.z) - ((ac.z * ab.y)))
            y = ((ac.z * ab.x) - ((ac.x * ab.z)))
            z = ((ac.x * ab.y) - ((ac.y * ab.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            direction = dest
            a = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x * -1)
            dest.y = (a.y * -1)
            dest.z = (a.z * -1)
            a0 = dest
            if ((((direction.x * a0.x) + ((direction.y * a0.y))) + ((direction.z * a0.z))) < 0):
                this1 = glm_Vec3Base()
                this1.x = 0
                this1.y = 0
                this1.z = 0
                dest = this1
                dest.x = (direction.x * -1)
                dest.y = (direction.y * -1)
                dest.z = (direction.z * -1)
                direction = dest
        elif (_g == 4):
            a = (vertices[3] if 3 < len(vertices) else None)
            b = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            da = dest
            a = (vertices[3] if 3 < len(vertices) else None)
            b = (vertices[1] if 1 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            db = dest
            a = (vertices[3] if 3 < len(vertices) else None)
            b = (vertices[2] if 2 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            dest.z = (a.z - b.z)
            dc = dest
            a = (vertices[3] if 3 < len(vertices) else None)
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            dest.x = (a.x * -1)
            dest.y = (a.y * -1)
            dest.z = (a.z * -1)
            d0 = dest
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            x = ((da.y * db.z) - ((da.z * db.y)))
            y = ((da.z * db.x) - ((da.x * db.z)))
            z = ((da.x * db.y) - ((da.y * db.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            abdNorm = dest
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            x = ((db.y * dc.z) - ((db.z * dc.y)))
            y = ((db.z * dc.x) - ((db.x * dc.z)))
            z = ((db.x * dc.y) - ((db.y * dc.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            bcdNorm = dest
            this1 = glm_Vec3Base()
            this1.x = 0
            this1.y = 0
            this1.z = 0
            dest = this1
            x = ((dc.y * da.z) - ((dc.z * da.y)))
            y = ((dc.z * da.x) - ((dc.x * da.z)))
            z = ((dc.x * da.y) - ((dc.y * da.x)))
            if (z is None):
                z = 0
            if (y is None):
                y = 0
            if (x is None):
                x = 0
            this1 = glm_Vec3Base()
            this1.x = x
            this1.y = y
            this1.z = z
            dest = this1
            cadNorm = dest
            if ((((abdNorm.x * d0.x) + ((abdNorm.y * d0.y))) + ((abdNorm.z * d0.z))) > 0):
                python_internal_ArrayImpl.remove(vertices,(vertices[2] if 2 < len(vertices) else None))
                direction = abdNorm
            elif ((((bcdNorm.x * d0.x) + ((bcdNorm.y * d0.y))) + ((bcdNorm.z * d0.z))) > 0):
                python_internal_ArrayImpl.remove(vertices,(vertices[0] if 0 < len(vertices) else None))
                direction = bcdNorm
            elif ((((cadNorm.x * d0.x) + ((cadNorm.y * d0.y))) + ((cadNorm.z * d0.z))) > 0):
                python_internal_ArrayImpl.remove(vertices,(vertices[1] if 1 < len(vertices) else None))
                direction = cadNorm
            else:
                return headbutt_threed__Headbutt_EvolveResult.FoundIntersection
        else:
            raise haxe_Exception.thrown((("Can't have simplex with " + Std.string(len(vertices))) + " verts!"))
        if headbutt_threed_Headbutt.addSupport(vertices,shapeA,shapeB,direction):
            return headbutt_threed__Headbutt_EvolveResult.StillEvolving
        else:
            return headbutt_threed__Headbutt_EvolveResult.NoIntersection

    @staticmethod
    def test(shapeA,shapeB):
        vertices = []
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        direction = this1
        result = headbutt_threed__Headbutt_EvolveResult.StillEvolving
        iterations = 0
        while ((iterations < headbutt_threed_Headbutt.MAX_ITERATIONS) and ((result == headbutt_threed__Headbutt_EvolveResult.StillEvolving))):
            result = headbutt_threed_Headbutt.evolveSimplex(vertices,shapeA,shapeB,direction)
            iterations = (iterations + 1)
        return headbutt_threed__TestResult_TestResult_Impl_.toBool(headbutt_threed__TestResult_TestResult_Impl_._new((result == headbutt_threed__Headbutt_EvolveResult.FoundIntersection),vertices,shapeA,shapeB))
headbutt_threed_Headbutt._hx_class = headbutt_threed_Headbutt


class headbutt_threed_Shape:
    _hx_class_name = "headbutt.threed.Shape"
    __slots__ = ()
    _hx_methods = ["get_centre", "support"]
headbutt_threed_Shape._hx_class = headbutt_threed_Shape


class headbutt_threed__TestResult__TestResult:
    _hx_class_name = "headbutt.threed._TestResult._TestResult"
    __slots__ = ("colliding", "simplex", "shapeA", "shapeB")
    _hx_fields = ["colliding", "simplex", "shapeA", "shapeB"]

    def __init__(self,colliding,simplex,shapeA,shapeB):
        self.colliding = colliding
        self.simplex = simplex
        self.shapeA = shapeA
        self.shapeB = shapeB

headbutt_threed__TestResult__TestResult._hx_class = headbutt_threed__TestResult__TestResult


class headbutt_threed__TestResult_TestResult_Impl_:
    _hx_class_name = "headbutt.threed._TestResult.TestResult_Impl_"
    __slots__ = ()
    _hx_statics = ["_new", "toBool"]

    @staticmethod
    def _new(colliding,simplex,shapeA,shapeB):
        this1 = headbutt_threed__TestResult__TestResult(colliding,simplex,shapeA,shapeB)
        return this1

    @staticmethod
    def toBool(this1):
        return this1.colliding
headbutt_threed__TestResult_TestResult_Impl_._hx_class = headbutt_threed__TestResult_TestResult_Impl_


class headbutt_threed_shapes_Box:
    _hx_class_name = "headbutt.threed.shapes.Box"
    __slots__ = ("vertices", "transform")
    _hx_fields = ["vertices", "transform"]
    _hx_methods = ["get_centre", "resize", "setTransform", "support"]
    _hx_interfaces = [headbutt_threed_Shape]

    def __init__(self,size):
        this1 = glm_Mat4Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._03 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._13 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        this1._23 = 0
        this1._30 = 0
        this1._31 = 0
        this1._32 = 0
        this1._33 = 0
        dest = this1
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        self.transform = dest
        this1 = [None]*8
        self.vertices = this1
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[0] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[1] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[2] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[3] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[4] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[5] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[6] = val
        this1 = self.vertices
        this2 = glm_Vec3Base()
        this2.x = 0
        this2.y = 0
        this2.z = 0
        val = this2
        this1[7] = val
        self.resize(size)

    def get_centre(self):
        x = self.transform._30
        y = self.transform._31
        z = self.transform._32
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        return this1

    def resize(self,size):
        self.vertices[0].x = (-0.5 * size.x)
        self.vertices[0].y = (-0.5 * size.y)
        self.vertices[0].z = (-0.5 * size.z)
        self.vertices[1].x = (0.5 * size.x)
        self.vertices[1].y = (-0.5 * size.y)
        self.vertices[1].z = (-0.5 * size.z)
        self.vertices[2].x = (0.5 * size.x)
        self.vertices[2].y = (0.5 * size.y)
        self.vertices[2].z = (-0.5 * size.z)
        self.vertices[3].x = (-0.5 * size.x)
        self.vertices[3].y = (0.5 * size.y)
        self.vertices[3].z = (-0.5 * size.z)
        self.vertices[4].x = (-0.5 * size.x)
        self.vertices[4].y = (-0.5 * size.y)
        self.vertices[4].z = (0.5 * size.z)
        self.vertices[5].x = (0.5 * size.x)
        self.vertices[5].y = (-0.5 * size.y)
        self.vertices[5].z = (0.5 * size.z)
        self.vertices[6].x = (0.5 * size.x)
        self.vertices[6].y = (0.5 * size.y)
        self.vertices[6].z = (0.5 * size.z)
        self.vertices[7].x = (-0.5 * size.x)
        self.vertices[7].y = (0.5 * size.y)
        self.vertices[7].z = (0.5 * size.z)

    def setTransform(self,position,rotation,scale):
        dest = self.transform
        x2 = (rotation.x + rotation.x)
        y2 = (rotation.y + rotation.y)
        z2 = (rotation.z + rotation.z)
        xx = (rotation.x * x2)
        xy = (rotation.x * y2)
        xz = (rotation.x * z2)
        yy = (rotation.y * y2)
        yz = (rotation.y * z2)
        zz = (rotation.z * z2)
        wx = (rotation.w * x2)
        wy = (rotation.w * y2)
        wz = (rotation.w * z2)
        dest._00 = (((1 - ((yy + zz)))) * scale.x)
        dest._01 = (((xy + wz)) * scale.x)
        dest._02 = (((xz - wy)) * scale.x)
        dest._03 = 0
        dest._10 = (((xy - wz)) * scale.y)
        dest._11 = (((1 - ((xx + zz)))) * scale.y)
        dest._12 = (((yz + wx)) * scale.y)
        dest._13 = 0
        dest._20 = (((xz + wy)) * scale.z)
        dest._21 = (((yz - wx)) * scale.z)
        dest._22 = (((1 - ((xx + yy)))) * scale.z)
        dest._23 = 0
        dest._30 = position.x
        dest._31 = position.y
        dest._32 = position.z
        dest._33 = 1
        self.transform = dest

    def support(self,direction):
        furthestDistance = Math.NEGATIVE_INFINITY
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        furthestVertex = this1
        x = 0
        y = 0
        z = 0
        w = 1
        if (w is None):
            w = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec4Base()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        vi = this1
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        vo = this1
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        vd = this1
        _g = 0
        _g1 = self.vertices
        while (_g < len(_g1)):
            v = _g1[_g]
            _g = (_g + 1)
            vi.x = v.x
            vi.y = v.y
            vi.z = v.z
            m = self.transform
            x = vi.x
            y = vi.y
            z = vi.z
            w = vi.w
            vo.x = ((((m._00 * x) + ((m._10 * y))) + ((m._20 * z))) + ((m._30 * w)))
            vo.y = ((((m._01 * x) + ((m._11 * y))) + ((m._21 * z))) + ((m._31 * w)))
            vo.z = ((((m._02 * x) + ((m._12 * y))) + ((m._22 * z))) + ((m._32 * w)))
            vo.w = ((((m._03 * x) + ((m._13 * y))) + ((m._23 * z))) + ((m._33 * w)))
            vd.x = (vo.x / vo.w)
            vd.y = (vo.y / vo.w)
            vd.z = (vo.z / vo.w)
            distance = (((vd.x * direction.x) + ((vd.y * direction.y))) + ((vd.z * direction.z)))
            if (distance > furthestDistance):
                furthestDistance = distance
                furthestVertex.x = vd.x
                furthestVertex.y = vd.y
                furthestVertex.z = vd.z
        return furthestVertex

headbutt_threed_shapes_Box._hx_class = headbutt_threed_shapes_Box


class headbutt_threed_shapes_Line:
    _hx_class_name = "headbutt.threed.shapes.Line"
    __slots__ = ("start", "end")
    _hx_fields = ["start", "end"]
    _hx_methods = ["get_centre", "support"]
    _hx_interfaces = [headbutt_threed_Shape]

    def __init__(self,start,end):
        self.start = start
        self.end = end

    def get_centre(self):
        x = (((self.start.x + self.end.x)) / 2)
        y = (((self.start.y + self.end.y)) / 2)
        z = (((self.start.z + self.end.z)) / 2)
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        return this1

    def support(self,direction):
        a = self.start
        a1 = self.end
        if ((((a.x * direction.x) + ((a.y * direction.y))) + ((a.z * direction.z))) > ((((a1.x * direction.x) + ((a1.y * direction.y))) + ((a1.z * direction.z))))):
            return self.start
        else:
            return self.end

headbutt_threed_shapes_Line._hx_class = headbutt_threed_shapes_Line


class headbutt_threed_shapes_Polyhedron:
    _hx_class_name = "headbutt.threed.shapes.Polyhedron"
    __slots__ = ("vertices", "transform")
    _hx_fields = ["vertices", "transform"]
    _hx_methods = ["get_centre", "setTransform", "support"]
    _hx_interfaces = [headbutt_threed_Shape]

    def __init__(self,vertices):
        this1 = glm_Mat4Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._03 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._13 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        this1._23 = 0
        this1._30 = 0
        this1._31 = 0
        this1._32 = 0
        this1._33 = 0
        dest = this1
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._30 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._31 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        dest._32 = 0
        dest._03 = 0
        dest._13 = 0
        dest._23 = 0
        dest._33 = 1
        self.transform = dest
        self.vertices = vertices

    def get_centre(self):
        x = self.transform._30
        y = self.transform._31
        z = self.transform._32
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        return this1

    def setTransform(self,position,rotation,scale):
        dest = self.transform
        x2 = (rotation.x + rotation.x)
        y2 = (rotation.y + rotation.y)
        z2 = (rotation.z + rotation.z)
        xx = (rotation.x * x2)
        xy = (rotation.x * y2)
        xz = (rotation.x * z2)
        yy = (rotation.y * y2)
        yz = (rotation.y * z2)
        zz = (rotation.z * z2)
        wx = (rotation.w * x2)
        wy = (rotation.w * y2)
        wz = (rotation.w * z2)
        dest._00 = (((1 - ((yy + zz)))) * scale.x)
        dest._01 = (((xy + wz)) * scale.x)
        dest._02 = (((xz - wy)) * scale.x)
        dest._03 = 0
        dest._10 = (((xy - wz)) * scale.y)
        dest._11 = (((1 - ((xx + zz)))) * scale.y)
        dest._12 = (((yz + wx)) * scale.y)
        dest._13 = 0
        dest._20 = (((xz + wy)) * scale.z)
        dest._21 = (((yz - wx)) * scale.z)
        dest._22 = (((1 - ((xx + yy)))) * scale.z)
        dest._23 = 0
        dest._30 = position.x
        dest._31 = position.y
        dest._32 = position.z
        dest._33 = 1
        self.transform = dest

    def support(self,direction):
        furthestDistance = Math.NEGATIVE_INFINITY
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        furthestVertex = this1
        x = 0
        y = 0
        z = 0
        w = 1
        if (w is None):
            w = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec4Base()
        this1.x = x
        this1.y = y
        this1.z = z
        this1.w = w
        vi = this1
        this1 = glm_Vec4Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        this1.w = 0
        vo = this1
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        vd = this1
        _g = 0
        _g1 = self.vertices
        while (_g < len(_g1)):
            v = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            vi.x = v.x
            vi.y = v.y
            vi.z = v.z
            m = self.transform
            x = vi.x
            y = vi.y
            z = vi.z
            w = vi.w
            vo.x = ((((m._00 * x) + ((m._10 * y))) + ((m._20 * z))) + ((m._30 * w)))
            vo.y = ((((m._01 * x) + ((m._11 * y))) + ((m._21 * z))) + ((m._31 * w)))
            vo.z = ((((m._02 * x) + ((m._12 * y))) + ((m._22 * z))) + ((m._32 * w)))
            vo.w = ((((m._03 * x) + ((m._13 * y))) + ((m._23 * z))) + ((m._33 * w)))
            vd.x = (vo.x / vo.w)
            vd.y = (vo.y / vo.w)
            vd.z = (vo.z / vo.w)
            distance = (((vd.x * direction.x) + ((vd.y * direction.y))) + ((vd.z * direction.z)))
            if (distance > furthestDistance):
                furthestDistance = distance
                furthestVertex.x = vd.x
                furthestVertex.y = vd.y
                furthestVertex.z = vd.z
        return furthestVertex

headbutt_threed_shapes_Polyhedron._hx_class = headbutt_threed_shapes_Polyhedron


class headbutt_threed_shapes_Sphere:
    _hx_class_name = "headbutt.threed.shapes.Sphere"
    __slots__ = ("position", "radius")
    _hx_fields = ["position", "radius"]
    _hx_methods = ["get_centre", "support"]
    _hx_interfaces = [headbutt_threed_Shape]

    def __init__(self,position,radius):
        self.position = position
        self.radius = radius

    def get_centre(self):
        return self.position

    def support(self,direction):
        src = self.position
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        dest.x = src.x
        dest.y = src.y
        dest.z = src.z
        c = dest
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        v = (((direction.x * direction.x) + ((direction.y * direction.y))) + ((direction.z * direction.z)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (direction.x * mult)
        dest.y = (direction.y * mult)
        dest.z = (direction.z * mult)
        d = dest
        s = self.radius
        _hx_local_0 = d
        _hx_local_1 = _hx_local_0.x
        _hx_local_0.x = (_hx_local_1 * s)
        _hx_local_0.x
        _hx_local_2 = d
        _hx_local_3 = _hx_local_2.y
        _hx_local_2.y = (_hx_local_3 * s)
        _hx_local_2.y
        _hx_local_4 = d
        _hx_local_5 = _hx_local_4.z
        _hx_local_4.z = (_hx_local_5 * s)
        _hx_local_4.z
        _hx_local_6 = c
        _hx_local_7 = _hx_local_6.x
        _hx_local_6.x = (_hx_local_7 + d.x)
        _hx_local_6.x
        _hx_local_8 = c
        _hx_local_9 = _hx_local_8.y
        _hx_local_8.y = (_hx_local_9 + d.y)
        _hx_local_8.y
        _hx_local_10 = c
        _hx_local_11 = _hx_local_10.z
        _hx_local_10.z = (_hx_local_11 + d.z)
        _hx_local_10.z
        return c

headbutt_threed_shapes_Sphere._hx_class = headbutt_threed_shapes_Sphere


class headbutt_twod__Headbutt_Edge:
    _hx_class_name = "headbutt.twod._Headbutt.Edge"
    __slots__ = ("distance", "normal", "index")
    _hx_fields = ["distance", "normal", "index"]

    def __init__(self,distance,normal,index):
        self.distance = distance
        self.normal = normal
        self.index = index

headbutt_twod__Headbutt_Edge._hx_class = headbutt_twod__Headbutt_Edge

class headbutt_twod__Headbutt_EvolveResult(Enum):
    __slots__ = ()
    _hx_class_name = "headbutt.twod._Headbutt.EvolveResult"
    _hx_constructs = ["NoIntersection", "FoundIntersection", "StillEvolving"]
headbutt_twod__Headbutt_EvolveResult.NoIntersection = headbutt_twod__Headbutt_EvolveResult("NoIntersection", 0, ())
headbutt_twod__Headbutt_EvolveResult.FoundIntersection = headbutt_twod__Headbutt_EvolveResult("FoundIntersection", 1, ())
headbutt_twod__Headbutt_EvolveResult.StillEvolving = headbutt_twod__Headbutt_EvolveResult("StillEvolving", 2, ())
headbutt_twod__Headbutt_EvolveResult._hx_class = headbutt_twod__Headbutt_EvolveResult

class headbutt_twod__Headbutt_PolygonWinding(Enum):
    __slots__ = ()
    _hx_class_name = "headbutt.twod._Headbutt.PolygonWinding"
    _hx_constructs = ["Clockwise", "CounterClockwise"]
headbutt_twod__Headbutt_PolygonWinding.Clockwise = headbutt_twod__Headbutt_PolygonWinding("Clockwise", 0, ())
headbutt_twod__Headbutt_PolygonWinding.CounterClockwise = headbutt_twod__Headbutt_PolygonWinding("CounterClockwise", 1, ())
headbutt_twod__Headbutt_PolygonWinding._hx_class = headbutt_twod__Headbutt_PolygonWinding


class headbutt_twod_Headbutt:
    _hx_class_name = "headbutt.twod.Headbutt"
    __slots__ = ()
    _hx_statics = ["MAX_ITERATIONS", "MAX_INTERSECTION_ITERATIONS", "INTERSECTION_EPSILON", "calculateSupport", "addSupport", "tripleProduct", "evolveSimplex", "test", "findClosestEdge", "intersect", "testAndIntersect"]

    def __init__(self):
        pass

    @staticmethod
    def calculateSupport(shapeA,shapeB,direction):
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = (direction.x * -1)
        dest.y = (direction.y * -1)
        oppositeDirection = dest
        src = shapeA.support(direction)
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = src.x
        dest.y = src.y
        newVertex = dest
        b = shapeB.support(oppositeDirection)
        _hx_local_0 = newVertex
        _hx_local_1 = _hx_local_0.x
        _hx_local_0.x = (_hx_local_1 - b.x)
        _hx_local_0.x
        _hx_local_2 = newVertex
        _hx_local_3 = _hx_local_2.y
        _hx_local_2.y = (_hx_local_3 - b.y)
        _hx_local_2.y
        return newVertex

    @staticmethod
    def addSupport(vertices,shapeA,shapeB,direction):
        newVertex = headbutt_twod_Headbutt.calculateSupport(shapeA,shapeB,direction)
        vertices.append(newVertex)
        return (((direction.x * newVertex.x) + ((direction.y * newVertex.y))) >= 0)

    @staticmethod
    def tripleProduct(a,b,c):
        x = a.x
        y = a.y
        z = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        A = this1
        x = b.x
        y = b.y
        z = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        B = this1
        x = c.x
        y = c.y
        z = 0
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        C = this1
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        x = ((A.y * B.z) - ((A.z * B.y)))
        y = ((A.z * B.x) - ((A.x * B.z)))
        z = ((A.x * B.y) - ((A.y * B.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest = this1
        first = dest
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        dest = this1
        x = ((first.y * C.z) - ((first.z * C.y)))
        y = ((first.z * C.x) - ((first.x * C.z)))
        z = ((first.x * C.y) - ((first.y * C.x)))
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        dest = this1
        second = dest
        x = second.x
        y = second.y
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    @staticmethod
    def evolveSimplex(vertices,shapeA,shapeB,direction):
        _g = len(vertices)
        if (_g == 0):
            a = shapeB.get_centre()
            b = shapeA.get_centre()
            direction.x = (a.x - b.x)
            direction.y = (a.y - b.y)
        elif (_g == 1):
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (direction.x * -1)
            dest.y = (direction.y * -1)
            direction = dest
        elif (_g == 2):
            a = (vertices[1] if 1 < len(vertices) else None)
            b = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            ab = dest
            a = (vertices[0] if 0 < len(vertices) else None)
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (a.x * -1)
            dest.y = (a.y * -1)
            a0 = dest
            direction = headbutt_twod_Headbutt.tripleProduct(ab,a0,ab)
        elif (_g == 3):
            a = (vertices[2] if 2 < len(vertices) else None)
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (a.x * -1)
            dest.y = (a.y * -1)
            c0 = dest
            a = (vertices[1] if 1 < len(vertices) else None)
            b = (vertices[2] if 2 < len(vertices) else None)
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            bc = dest
            a = (vertices[0] if 0 < len(vertices) else None)
            b = (vertices[2] if 2 < len(vertices) else None)
            this1 = glm_Vec2Base()
            this1.x = 0
            this1.y = 0
            dest = this1
            dest.x = (a.x - b.x)
            dest.y = (a.y - b.y)
            ca = dest
            bcNorm = headbutt_twod_Headbutt.tripleProduct(ca,bc,bc)
            caNorm = headbutt_twod_Headbutt.tripleProduct(bc,ca,ca)
            if (((bcNorm.x * c0.x) + ((bcNorm.y * c0.y))) > 0):
                python_internal_ArrayImpl.remove(vertices,(vertices[0] if 0 < len(vertices) else None))
                direction = bcNorm
            elif (((caNorm.x * c0.x) + ((caNorm.y * c0.y))) > 0):
                python_internal_ArrayImpl.remove(vertices,(vertices[1] if 1 < len(vertices) else None))
                direction = caNorm
            else:
                return headbutt_twod__Headbutt_EvolveResult.FoundIntersection
        else:
            raise haxe_Exception.thrown((("Can't have simplex with " + Std.string(len(vertices))) + " verts!"))
        if headbutt_twod_Headbutt.addSupport(vertices,shapeA,shapeB,direction):
            return headbutt_twod__Headbutt_EvolveResult.StillEvolving
        else:
            return headbutt_twod__Headbutt_EvolveResult.NoIntersection

    @staticmethod
    def test(shapeA,shapeB):
        vertices = []
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        direction = this1
        result = headbutt_twod__Headbutt_EvolveResult.StillEvolving
        iterations = 0
        while ((iterations < headbutt_twod_Headbutt.MAX_ITERATIONS) and ((result == headbutt_twod__Headbutt_EvolveResult.StillEvolving))):
            result = headbutt_twod_Headbutt.evolveSimplex(vertices,shapeA,shapeB,direction)
            iterations = (iterations + 1)
        return headbutt_twod__TestResult_TestResult_Impl_._new((result == headbutt_twod__Headbutt_EvolveResult.FoundIntersection),vertices,shapeA,shapeB)

    @staticmethod
    def findClosestEdge(vertices,winding):
        closestDistance = Math.POSITIVE_INFINITY
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        closestNormal = this1
        closestIndex = 0
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        line = this1
        _g = 0
        _g1 = len(vertices)
        while (_g < _g1):
            i = _g
            _g = (_g + 1)
            j = (i + 1)
            if (j >= len(vertices)):
                j = 0
            src = (vertices[j] if j >= 0 and j < len(vertices) else None)
            line.x = src.x
            line.y = src.y
            b = (vertices[i] if i >= 0 and i < len(vertices) else None)
            _hx_local_0 = line
            _hx_local_1 = _hx_local_0.x
            _hx_local_0.x = (_hx_local_1 - b.x)
            _hx_local_0.x
            _hx_local_2 = line
            _hx_local_3 = _hx_local_2.y
            _hx_local_2.y = (_hx_local_3 - b.y)
            _hx_local_2.y
            norm = None
            norm1 = winding.index
            if (norm1 == 0):
                x = line.y
                y = -line.x
                if (y is None):
                    y = 0
                if (x is None):
                    x = 0
                this1 = glm_Vec2Base()
                this1.x = x
                this1.y = y
                norm = this1
            elif (norm1 == 1):
                x1 = -line.y
                y1 = line.x
                if (y1 is None):
                    y1 = 0
                if (x1 is None):
                    x1 = 0
                this2 = glm_Vec2Base()
                this2.x = x1
                this2.y = y1
                norm = this2
            else:
                pass
            v = ((norm.x * norm.x) + ((norm.y * norm.y)))
            length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
            mult = 0
            if (length >= glm_GLM.EPSILON):
                mult = (1 / length)
            _hx_local_4 = norm
            _hx_local_5 = _hx_local_4.x
            _hx_local_4.x = (_hx_local_5 * mult)
            _hx_local_4.x
            _hx_local_6 = norm
            _hx_local_7 = _hx_local_6.y
            _hx_local_6.y = (_hx_local_7 * mult)
            _hx_local_6.y
            b1 = (vertices[i] if i >= 0 and i < len(vertices) else None)
            dist = ((norm.x * b1.x) + ((norm.y * b1.y)))
            if (dist < closestDistance):
                closestDistance = dist
                closestNormal = norm
                closestIndex = j
        return headbutt_twod__Headbutt_Edge(closestDistance,closestNormal,closestIndex)

    @staticmethod
    def intersect(testResult):
        if (not testResult.colliding):
            return None
        e0 = ((((testResult.simplex[1] if 1 < len(testResult.simplex) else None).x - (testResult.simplex[0] if 0 < len(testResult.simplex) else None).x)) * (((testResult.simplex[1] if 1 < len(testResult.simplex) else None).y + (testResult.simplex[0] if 0 < len(testResult.simplex) else None).y)))
        e1 = ((((testResult.simplex[2] if 2 < len(testResult.simplex) else None).x - (testResult.simplex[1] if 1 < len(testResult.simplex) else None).x)) * (((testResult.simplex[2] if 2 < len(testResult.simplex) else None).y + (testResult.simplex[1] if 1 < len(testResult.simplex) else None).y)))
        e2 = ((((testResult.simplex[0] if 0 < len(testResult.simplex) else None).x - (testResult.simplex[2] if 2 < len(testResult.simplex) else None).x)) * (((testResult.simplex[0] if 0 < len(testResult.simplex) else None).y + (testResult.simplex[2] if 2 < len(testResult.simplex) else None).y)))
        winding = (headbutt_twod__Headbutt_PolygonWinding.Clockwise if ((((e0 + e1) + e2) >= 0)) else headbutt_twod__Headbutt_PolygonWinding.CounterClockwise)
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        intersection = this1
        _g = 0
        _g1 = headbutt_twod_Headbutt.MAX_INTERSECTION_ITERATIONS
        while (_g < _g1):
            _ = _g
            _g = (_g + 1)
            edge = headbutt_twod_Headbutt.findClosestEdge(testResult.simplex,winding)
            support = headbutt_twod_Headbutt.calculateSupport(testResult.shapeA,testResult.shapeB,edge.normal)
            b = edge.normal
            distance = ((support.x * b.x) + ((support.y * b.y)))
            src = edge.normal
            intersection.x = src.x
            intersection.y = src.y
            _hx_local_0 = intersection
            _hx_local_1 = _hx_local_0.x
            _hx_local_0.x = (_hx_local_1 * distance)
            _hx_local_0.x
            _hx_local_2 = intersection
            _hx_local_3 = _hx_local_2.y
            _hx_local_2.y = (_hx_local_3 * distance)
            _hx_local_2.y
            if (Reflect.field(Math,"fabs")((distance - edge.distance)) <= headbutt_twod_Headbutt.INTERSECTION_EPSILON):
                return headbutt_twod_IntersectResult(intersection,testResult.shapeA,testResult.shapeB)
            else:
                pos = edge.index
                testResult.simplex.insert(pos, support)
        return headbutt_twod_IntersectResult(intersection,testResult.shapeA,testResult.shapeB)

    @staticmethod
    def testAndIntersect(shapeA,shapeB):
        return headbutt_twod_Headbutt.intersect(headbutt_twod_Headbutt.test(shapeA,shapeB))
headbutt_twod_Headbutt._hx_class = headbutt_twod_Headbutt


class headbutt_twod_IntersectResult:
    _hx_class_name = "headbutt.twod.IntersectResult"
    __slots__ = ("intersection", "shapeA", "shapeB", "normal", "overlapDistance")
    _hx_fields = ["intersection", "shapeA", "shapeB", "normal", "overlapDistance"]

    def __init__(self,intersection,shapeA,shapeB):
        self.intersection = intersection
        self.shapeA = shapeA
        self.shapeB = shapeB
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        v = ((intersection.x * intersection.x) + ((intersection.y * intersection.y)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (intersection.x * mult)
        dest.y = (intersection.y * mult)
        self.normal = dest
        v = ((intersection.x * intersection.x) + ((intersection.y * intersection.y)))
        self.overlapDistance = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))

headbutt_twod_IntersectResult._hx_class = headbutt_twod_IntersectResult


class headbutt_twod_Shape:
    _hx_class_name = "headbutt.twod.Shape"
    __slots__ = ()
    _hx_methods = ["get_centre", "support"]
headbutt_twod_Shape._hx_class = headbutt_twod_Shape


class headbutt_twod__TestResult__TestResult:
    _hx_class_name = "headbutt.twod._TestResult._TestResult"
    __slots__ = ("colliding", "simplex", "shapeA", "shapeB")
    _hx_fields = ["colliding", "simplex", "shapeA", "shapeB"]

    def __init__(self,colliding,simplex,shapeA,shapeB):
        self.colliding = colliding
        self.simplex = simplex
        self.shapeA = shapeA
        self.shapeB = shapeB

headbutt_twod__TestResult__TestResult._hx_class = headbutt_twod__TestResult__TestResult


class headbutt_twod__TestResult_TestResult_Impl_:
    _hx_class_name = "headbutt.twod._TestResult.TestResult_Impl_"
    __slots__ = ()
    _hx_statics = ["_new", "toBool"]

    @staticmethod
    def _new(colliding,simplex,shapeA,shapeB):
        this1 = headbutt_twod__TestResult__TestResult(colliding,simplex,shapeA,shapeB)
        return this1

    @staticmethod
    def toBool(this1):
        return this1.colliding
headbutt_twod__TestResult_TestResult_Impl_._hx_class = headbutt_twod__TestResult_TestResult_Impl_


class headbutt_twod_shapes_Circle:
    _hx_class_name = "headbutt.twod.shapes.Circle"
    __slots__ = ("position", "radius")
    _hx_fields = ["position", "radius"]
    _hx_methods = ["get_centre", "support"]
    _hx_interfaces = [headbutt_twod_Shape]

    def __init__(self,position,radius):
        self.position = position
        self.radius = radius

    def get_centre(self):
        return self.position

    def support(self,direction):
        src = self.position
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        dest.x = src.x
        dest.y = src.y
        c = dest
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        dest = this1
        v = ((direction.x * direction.x) + ((direction.y * direction.y)))
        length = (Math.NaN if ((v < 0)) else python_lib_Math.sqrt(v))
        mult = 0
        if (length >= glm_GLM.EPSILON):
            mult = (1 / length)
        dest.x = (direction.x * mult)
        dest.y = (direction.y * mult)
        d = dest
        s = self.radius
        _hx_local_0 = d
        _hx_local_1 = _hx_local_0.x
        _hx_local_0.x = (_hx_local_1 * s)
        _hx_local_0.x
        _hx_local_2 = d
        _hx_local_3 = _hx_local_2.y
        _hx_local_2.y = (_hx_local_3 * s)
        _hx_local_2.y
        _hx_local_4 = c
        _hx_local_5 = _hx_local_4.x
        _hx_local_4.x = (_hx_local_5 + d.x)
        _hx_local_4.x
        _hx_local_6 = c
        _hx_local_7 = _hx_local_6.y
        _hx_local_6.y = (_hx_local_7 + d.y)
        _hx_local_6.y
        return c

headbutt_twod_shapes_Circle._hx_class = headbutt_twod_shapes_Circle


class headbutt_twod_shapes_Line:
    _hx_class_name = "headbutt.twod.shapes.Line"
    __slots__ = ("start", "end")
    _hx_fields = ["start", "end"]
    _hx_methods = ["get_centre", "support"]
    _hx_interfaces = [headbutt_twod_Shape]

    def __init__(self,start,end):
        self.start = start
        self.end = end

    def get_centre(self):
        x = (((self.start.x + self.end.x)) / 2)
        y = (((self.start.y + self.end.y)) / 2)
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    def support(self,direction):
        a = self.start
        a1 = self.end
        if (((a.x * direction.x) + ((a.y * direction.y))) > (((a1.x * direction.x) + ((a1.y * direction.y))))):
            return self.start
        else:
            return self.end

headbutt_twod_shapes_Line._hx_class = headbutt_twod_shapes_Line


class headbutt_twod_shapes_Polygon:
    _hx_class_name = "headbutt.twod.shapes.Polygon"
    __slots__ = ("vertices", "transform")
    _hx_fields = ["vertices", "transform"]
    _hx_methods = ["get_centre", "setTransform", "support"]
    _hx_interfaces = [headbutt_twod_Shape]

    def __init__(self,vertices):
        this1 = glm_Mat3Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        dest = this1
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        self.transform = dest
        self.vertices = vertices

    def get_centre(self):
        x = self.transform._20
        y = self.transform._21
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    def setTransform(self,position,rotation,scale):
        c = (Math.NaN if (((rotation == Math.POSITIVE_INFINITY) or ((rotation == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(rotation))
        s = (Math.NaN if (((rotation == Math.POSITIVE_INFINITY) or ((rotation == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(rotation))
        self.transform._00 = (scale.x * c)
        self.transform._10 = -s
        self.transform._20 = position.x
        self.transform._01 = (scale.x * s)
        self.transform._11 = (scale.y * c)
        self.transform._21 = position.y
        self.transform._02 = 0
        self.transform._12 = 0
        self.transform._22 = 1

    def support(self,direction):
        furthestDistance = Math.NEGATIVE_INFINITY
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        furthestVertex = this1
        x = 0
        y = 0
        z = 1
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        vi = this1
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        vo = this1
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        vd = this1
        _g = 0
        _g1 = self.vertices
        while (_g < len(_g1)):
            v = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            vi.x = v.x
            vi.y = v.y
            m = self.transform
            x = vi.x
            y = vi.y
            z = vi.z
            vo.x = (((m._00 * x) + ((m._10 * y))) + ((m._20 * z)))
            vo.y = (((m._01 * x) + ((m._11 * y))) + ((m._21 * z)))
            vo.z = (((m._02 * x) + ((m._12 * y))) + ((m._22 * z)))
            vd.x = (vo.x / vo.z)
            vd.y = (vo.y / vo.z)
            distance = ((vd.x * direction.x) + ((vd.y * direction.y)))
            if (distance > furthestDistance):
                furthestDistance = distance
                furthestVertex.x = vd.x
                furthestVertex.y = vd.y
        return furthestVertex

headbutt_twod_shapes_Polygon._hx_class = headbutt_twod_shapes_Polygon


class headbutt_twod_shapes_Rectangle:
    _hx_class_name = "headbutt.twod.shapes.Rectangle"
    __slots__ = ("vertices", "transform")
    _hx_fields = ["vertices", "transform"]
    _hx_methods = ["get_centre", "resize", "setTransform", "support"]
    _hx_interfaces = [headbutt_twod_Shape]

    def __init__(self,size):
        this1 = glm_Mat3Base()
        this1._00 = 0
        this1._01 = 0
        this1._02 = 0
        this1._10 = 0
        this1._11 = 0
        this1._12 = 0
        this1._20 = 0
        this1._21 = 0
        this1._22 = 0
        dest = this1
        dest._00 = 1
        dest._10 = 0
        dest._20 = 0
        dest._01 = 0
        dest._11 = 1
        dest._21 = 0
        dest._02 = 0
        dest._12 = 0
        dest._22 = 1
        self.transform = dest
        this1 = [None]*4
        self.vertices = this1
        this1 = self.vertices
        this2 = glm_Vec2Base()
        this2.x = 0
        this2.y = 0
        val = this2
        this1[0] = val
        this1 = self.vertices
        this2 = glm_Vec2Base()
        this2.x = 0
        this2.y = 0
        val = this2
        this1[1] = val
        this1 = self.vertices
        this2 = glm_Vec2Base()
        this2.x = 0
        this2.y = 0
        val = this2
        this1[2] = val
        this1 = self.vertices
        this2 = glm_Vec2Base()
        this2.x = 0
        this2.y = 0
        val = this2
        this1[3] = val
        self.resize(size)

    def get_centre(self):
        x = self.transform._20
        y = self.transform._21
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec2Base()
        this1.x = x
        this1.y = y
        return this1

    def resize(self,size):
        self.vertices[0].x = (-0.5 * size.x)
        self.vertices[0].y = (-0.5 * size.y)
        self.vertices[1].x = (0.5 * size.x)
        self.vertices[1].y = (-0.5 * size.y)
        self.vertices[2].x = (0.5 * size.x)
        self.vertices[2].y = (0.5 * size.y)
        self.vertices[3].x = (-0.5 * size.x)
        self.vertices[3].y = (0.5 * size.y)

    def setTransform(self,position,rotation,scale):
        c = (Math.NaN if (((rotation == Math.POSITIVE_INFINITY) or ((rotation == Math.NEGATIVE_INFINITY)))) else python_lib_Math.cos(rotation))
        s = (Math.NaN if (((rotation == Math.POSITIVE_INFINITY) or ((rotation == Math.NEGATIVE_INFINITY)))) else python_lib_Math.sin(rotation))
        self.transform._00 = (scale.x * c)
        self.transform._10 = -s
        self.transform._20 = position.x
        self.transform._01 = (scale.x * s)
        self.transform._11 = (scale.y * c)
        self.transform._21 = position.y
        self.transform._02 = 0
        self.transform._12 = 0
        self.transform._22 = 1

    def support(self,direction):
        furthestDistance = Math.NEGATIVE_INFINITY
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        furthestVertex = this1
        x = 0
        y = 0
        z = 1
        if (z is None):
            z = 0
        if (y is None):
            y = 0
        if (x is None):
            x = 0
        this1 = glm_Vec3Base()
        this1.x = x
        this1.y = y
        this1.z = z
        vi = this1
        this1 = glm_Vec3Base()
        this1.x = 0
        this1.y = 0
        this1.z = 0
        vo = this1
        this1 = glm_Vec2Base()
        this1.x = 0
        this1.y = 0
        vd = this1
        _g = 0
        _g1 = self.vertices
        while (_g < len(_g1)):
            v = _g1[_g]
            _g = (_g + 1)
            vi.x = v.x
            vi.y = v.y
            m = self.transform
            x = vi.x
            y = vi.y
            z = vi.z
            vo.x = (((m._00 * x) + ((m._10 * y))) + ((m._20 * z)))
            vo.y = (((m._01 * x) + ((m._11 * y))) + ((m._21 * z)))
            vo.z = (((m._02 * x) + ((m._12 * y))) + ((m._22 * z)))
            vd.x = (vo.x / vo.z)
            vd.y = (vo.y / vo.z)
            distance = ((vd.x * direction.x) + ((vd.y * direction.y)))
            if (distance > furthestDistance):
                furthestDistance = distance
                furthestVertex.x = vd.x
                furthestVertex.y = vd.y
        return furthestVertex

headbutt_twod_shapes_Rectangle._hx_class = headbutt_twod_shapes_Rectangle


class promhx_base_AsyncBase:
    _hx_class_name = "promhx.base.AsyncBase"
    __slots__ = ("_val", "_resolved", "_fulfilled", "_pending", "_update", "_error", "_errored", "_errorMap", "_errorVal", "_errorPending")
    _hx_fields = ["_val", "_resolved", "_fulfilled", "_pending", "_update", "_error", "_errored", "_errorMap", "_errorVal", "_errorPending"]
    _hx_methods = ["catchError", "errorThen", "isResolved", "isErrored", "isErrorHandled", "isErrorPending", "isFulfilled", "isPending", "handleResolve", "_resolve", "handleError", "_handleError", "then", "unlink", "isLinked"]
    _hx_statics = ["link", "immediateLinkUpdate", "linkAll", "pipeLink", "allResolved", "allFulfilled"]

    def __init__(self,d = None):
        self._errorVal = None
        self._errorMap = None
        self._val = None
        self._resolved = False
        self._pending = False
        self._errorPending = False
        self._fulfilled = False
        self._update = []
        self._error = []
        self._errored = False
        if (d is not None):
            next = self
            def _hx_local_0(x):
                return x
            f = _hx_local_0
            _this = d._update
            def _hx_local_1(x):
                next.handleResolve(f(x))
            _this.append(_hx_AnonObject({'_hx_async': next, 'linkf': _hx_local_1}))
            promhx_base_AsyncBase.immediateLinkUpdate(d,next,f)

    def catchError(self,f):
        _this = self._error
        _this.append(f)
        return self

    def errorThen(self,f):
        self._errorMap = f
        return self

    def isResolved(self):
        return self._resolved

    def isErrored(self):
        return self._errored

    def isErrorHandled(self):
        return (len(self._error) > 0)

    def isErrorPending(self):
        return self._errorPending

    def isFulfilled(self):
        return self._fulfilled

    def isPending(self):
        return self._pending

    def handleResolve(self,val):
        self._resolve(val)

    def _resolve(self,val):
        _gthis = self
        if self._pending:
            _g = self._resolve
            val1 = val
            def _hx_local_0():
                _g(val1)
            tmp = _hx_local_0
            promhx_base_EventLoop.queue.add(tmp)
            promhx_base_EventLoop.continueOnNextLoop()
        else:
            self._resolved = True
            self._pending = True
            def _hx_local_2():
                _gthis._val = val
                _g = 0
                _g1 = _gthis._update
                while (_g < len(_g1)):
                    up = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                    _g = (_g + 1)
                    try:
                        up.linkf(val)
                    except BaseException as _g2:
                        None
                        e = haxe_Exception.caught(_g2).unwrap()
                        up._hx_async.handleError(e)
                _gthis._fulfilled = True
                _gthis._pending = False
            promhx_base_EventLoop.queue.add(_hx_local_2)
            promhx_base_EventLoop.continueOnNextLoop()

    def handleError(self,error):
        self._handleError(error)

    def _handleError(self,error):
        _gthis = self
        def _hx_local_2(e):
            if (len(_gthis._error) > 0):
                _g = 0
                _g1 = _gthis._error
                while (_g < len(_g1)):
                    ef = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                    _g = (_g + 1)
                    ef(e)
            elif (len(_gthis._update) > 0):
                _g = 0
                _g1 = _gthis._update
                while (_g < len(_g1)):
                    up = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
                    _g = (_g + 1)
                    up._hx_async.handleError(e)
            else:
                raise haxe_Exception.thrown(e)
            _gthis._errorPending = False
        update_errors = _hx_local_2
        if (not self._errorPending):
            self._errorPending = True
            self._errored = True
            self._errorVal = error
            def _hx_local_3():
                if (_gthis._errorMap is not None):
                    try:
                        _gthis._resolve(_gthis._errorMap(error))
                    except BaseException as _g:
                        None
                        e = haxe_Exception.caught(_g).unwrap()
                        update_errors(e)
                else:
                    update_errors(error)
            promhx_base_EventLoop.queue.add(_hx_local_3)
            promhx_base_EventLoop.continueOnNextLoop()

    def then(self,f):
        ret = promhx_base_AsyncBase(None)
        next = ret
        f1 = f
        _this = self._update
        def _hx_local_0(x):
            next.handleResolve(f1(x))
        _this.append(_hx_AnonObject({'_hx_async': next, 'linkf': _hx_local_0}))
        promhx_base_AsyncBase.immediateLinkUpdate(self,next,f1)
        return ret

    def unlink(self,to):
        _gthis = self
        def _hx_local_1():
            def _hx_local_0(x):
                return (x._hx_async != to)
            tmp = list(filter(_hx_local_0,_gthis._update))
            _gthis._update = tmp
        promhx_base_EventLoop.queue.add(_hx_local_1)
        promhx_base_EventLoop.continueOnNextLoop()

    def isLinked(self,to):
        updated = False
        _g = 0
        _g1 = self._update
        while (_g < len(_g1)):
            u = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            if (u._hx_async == to):
                return True
        return updated

    @staticmethod
    def link(current,next,f):
        _this = current._update
        def _hx_local_0(x):
            next.handleResolve(f(x))
        _this.append(_hx_AnonObject({'_hx_async': next, 'linkf': _hx_local_0}))
        promhx_base_AsyncBase.immediateLinkUpdate(current,next,f)

    @staticmethod
    def immediateLinkUpdate(current,next,f):
        if ((current._errored and (not current._errorPending)) and ((len(current._error) <= 0))):
            next.handleError(current._errorVal)
        if (current._resolved and (not current._pending)):
            try:
                next.handleResolve(f(current._val))
            except BaseException as _g:
                None
                e = haxe_Exception.caught(_g).unwrap()
                next.handleError(e)

    @staticmethod
    def linkAll(all,next):
        def _hx_local_0(arr,current,v):
            if ((len(arr) == 0) or promhx_base_AsyncBase.allFulfilled(arr)):
                _g = []
                a = HxOverrides.iterator(all)
                while a.hasNext():
                    a1 = a.next()
                    x = (v if ((a1 == current)) else a1._val)
                    _g.append(x)
                vals = _g
                next.handleResolve(vals)
        cthen = _hx_local_0
        a = HxOverrides.iterator(all)
        while a.hasNext():
            a1 = a.next()
            _this = a1._update
            next1 = next
            _g = [cthen]
            _g1 = []
            a2 = HxOverrides.iterator(all)
            while a2.hasNext():
                a21 = a2.next()
                if (a21 != a1):
                    _g1.append(a21)
            def _hx_local_2(_g,current,arr):
                def _hx_local_1(v):
                    (_g[0] if 0 < len(_g) else None)((arr[0] if 0 < len(arr) else None),(current[0] if 0 < len(current) else None),v)
                return _hx_local_1
            x = _hx_AnonObject({'_hx_async': next1, 'linkf': _hx_local_2(_g,[a1],[_g1])})
            _this.append(x)
        if promhx_base_AsyncBase.allFulfilled(all):
            next1 = next
            _g1 = []
            a = HxOverrides.iterator(all)
            while a.hasNext():
                a1 = a.next()
                x = a1._val
                _g1.append(x)
            next1.handleResolve(_g1)

    @staticmethod
    def pipeLink(current,ret,f):
        linked = False
        def _hx_local_1(x):
            nonlocal linked
            if (not linked):
                linked = True
                pipe_ret = f(x)
                _this = pipe_ret._update
                _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': ret.handleResolve}))
                def _hx_local_0(x):
                    return x
                promhx_base_AsyncBase.immediateLinkUpdate(pipe_ret,ret,_hx_local_0)
        linkf = _hx_local_1
        _this = current._update
        _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': linkf}))
        if (current._resolved and (not current._pending)):
            try:
                linkf(current._val)
            except BaseException as _g:
                None
                e = haxe_Exception.caught(_g).unwrap()
                ret.handleError(e)

    @staticmethod
    def allResolved(_hx_as):
        a = HxOverrides.iterator(_hx_as)
        while a.hasNext():
            a1 = a.next()
            if (not a1._resolved):
                return False
        return True

    @staticmethod
    def allFulfilled(_hx_as):
        a = HxOverrides.iterator(_hx_as)
        while a.hasNext():
            a1 = a.next()
            if (not a1._fulfilled):
                return False
        return True

promhx_base_AsyncBase._hx_class = promhx_base_AsyncBase


class promhx_Deferred(promhx_base_AsyncBase):
    _hx_class_name = "promhx.Deferred"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["resolve", "throwError", "promise", "stream", "publicStream"]
    _hx_statics = []
    _hx_interfaces = []
    _hx_super = promhx_base_AsyncBase


    def __init__(self):
        super().__init__()

    def resolve(self,val):
        self.handleResolve(val)

    def throwError(self,e):
        self.handleError(e)

    def promise(self):
        return promhx_Promise(self)

    def stream(self):
        return promhx_Stream(self)

    def publicStream(self):
        return promhx_PublicStream(self)

promhx_Deferred._hx_class = promhx_Deferred


class promhx_Promise(promhx_base_AsyncBase):
    _hx_class_name = "promhx.Promise"
    __slots__ = ("_rejected",)
    _hx_fields = ["_rejected"]
    _hx_methods = ["isRejected", "reject", "handleResolve", "then", "unlink", "handleError", "pipe", "errorPipe"]
    _hx_statics = ["whenAll", "promise"]
    _hx_interfaces = []
    _hx_super = promhx_base_AsyncBase


    def __init__(self,d = None):
        self._rejected = None
        super().__init__(d)
        self._rejected = False

    def isRejected(self):
        return self._rejected

    def reject(self,e):
        self._rejected = True
        self.handleError(e)

    def handleResolve(self,val):
        if self._resolved:
            msg = "Promise has already been resolved"
            raise haxe_Exception.thrown(promhx_error_PromiseError.AlreadyResolved(msg))
        self._resolve(val)

    def then(self,f):
        ret = promhx_Promise(None)
        next = ret
        f1 = f
        _this = self._update
        def _hx_local_0(x):
            next.handleResolve(f1(x))
        _this.append(_hx_AnonObject({'_hx_async': next, 'linkf': _hx_local_0}))
        promhx_base_AsyncBase.immediateLinkUpdate(self,next,f1)
        return ret

    def unlink(self,to):
        _gthis = self
        def _hx_local_1():
            if (not _gthis._fulfilled):
                msg = "Downstream Promise is not fullfilled"
                _gthis.handleError(promhx_error_PromiseError.DownstreamNotFullfilled(msg))
            else:
                def _hx_local_0(x):
                    return (x._hx_async != to)
                tmp = list(filter(_hx_local_0,_gthis._update))
                _gthis._update = tmp
        promhx_base_EventLoop.queue.add(_hx_local_1)
        promhx_base_EventLoop.continueOnNextLoop()

    def handleError(self,error):
        self._rejected = True
        self._handleError(error)

    def pipe(self,f):
        ret = promhx_Promise(None)
        ret1 = ret
        f1 = f
        linked = False
        def _hx_local_1(x):
            nonlocal linked
            if (not linked):
                linked = True
                pipe_ret = f1(x)
                _this = pipe_ret._update
                _this.append(_hx_AnonObject({'_hx_async': ret1, 'linkf': ret1.handleResolve}))
                def _hx_local_0(x):
                    return x
                promhx_base_AsyncBase.immediateLinkUpdate(pipe_ret,ret1,_hx_local_0)
        linkf = _hx_local_1
        _this = self._update
        _this.append(_hx_AnonObject({'_hx_async': ret1, 'linkf': linkf}))
        if (self._resolved and (not self._pending)):
            try:
                linkf(self._val)
            except BaseException as _g:
                None
                e = haxe_Exception.caught(_g).unwrap()
                ret1.handleError(e)
        return ret

    def errorPipe(self,f):
        ret = promhx_Promise()
        def _hx_local_0(e):
            piped = f(e)
            piped.then(ret._resolve)
        self.catchError(_hx_local_0)
        self.then(ret._resolve)
        return ret

    @staticmethod
    def whenAll(itb):
        ret = promhx_Promise(None)
        all = itb
        next = ret
        def _hx_local_0(arr,current,v):
            if ((len(arr) == 0) or promhx_base_AsyncBase.allFulfilled(arr)):
                _g = []
                a = HxOverrides.iterator(all)
                while a.hasNext():
                    a1 = a.next()
                    x = (v if ((a1 == current)) else a1._val)
                    _g.append(x)
                vals = _g
                next.handleResolve(vals)
        cthen = _hx_local_0
        a = HxOverrides.iterator(all)
        while a.hasNext():
            a1 = a.next()
            _this = a1._update
            next1 = next
            _g = [cthen]
            _g1 = []
            a2 = HxOverrides.iterator(all)
            while a2.hasNext():
                a21 = a2.next()
                if (a21 != a1):
                    _g1.append(a21)
            def _hx_local_2(current,arr,_g):
                def _hx_local_1(v):
                    (_g[0] if 0 < len(_g) else None)((arr[0] if 0 < len(arr) else None),(current[0] if 0 < len(current) else None),v)
                return _hx_local_1
            x = _hx_AnonObject({'_hx_async': next1, 'linkf': _hx_local_2([a1],[_g1],_g)})
            _this.append(x)
        if promhx_base_AsyncBase.allFulfilled(all):
            next1 = next
            _g1 = []
            a = HxOverrides.iterator(all)
            while a.hasNext():
                a1 = a.next()
                x = a1._val
                _g1.append(x)
            next1.handleResolve(_g1)
        return ret

    @staticmethod
    def promise(_val):
        ret = promhx_Promise()
        ret.handleResolve(_val)
        return ret

promhx_Promise._hx_class = promhx_Promise


class promhx_Stream(promhx_base_AsyncBase):
    _hx_class_name = "promhx.Stream"
    __slots__ = ("deferred", "_pause", "_end", "_end_promise")
    _hx_fields = ["deferred", "_pause", "_end", "_end_promise"]
    _hx_methods = ["then", "detachStream", "first", "handleResolve", "pause", "pipe", "errorPipe", "handleEnd", "end", "endThen", "filter", "concat", "merge"]
    _hx_statics = ["foreach", "wheneverAll", "concatAll", "mergeAll", "stream"]
    _hx_interfaces = []
    _hx_super = promhx_base_AsyncBase


    def __init__(self,d = None):
        self._end_promise = None
        self._end = None
        self._pause = None
        self.deferred = None
        super().__init__(d)
        self._end_promise = promhx_Promise()

    def then(self,f):
        ret = promhx_Stream(None)
        next = ret
        f1 = f
        _this = self._update
        def _hx_local_0(x):
            next.handleResolve(f1(x))
        _this.append(_hx_AnonObject({'_hx_async': next, 'linkf': _hx_local_0}))
        promhx_base_AsyncBase.immediateLinkUpdate(self,next,f1)
        _this = self._end_promise._update
        def _hx_local_1(x):
            ret.end()
        x = _hx_AnonObject({'_hx_async': ret._end_promise, 'linkf': _hx_local_1})
        _this.append(x)
        return ret

    def detachStream(self,_hx_str):
        filtered = []
        removed = False
        _g = 0
        _g1 = self._update
        while (_g < len(_g1)):
            u = (_g1[_g] if _g >= 0 and _g < len(_g1) else None)
            _g = (_g + 1)
            if (u._hx_async == _hx_str):
                def _hx_local_1(x):
                    return (x._hx_async != _hx_str._end_promise)
                tmp = list(filter(_hx_local_1,self._end_promise._update))
                self._end_promise._update = tmp
                removed = True
            else:
                filtered.append(u)
        self._update = filtered
        return removed

    def first(self):
        s = promhx_Promise(None)
        def _hx_local_0(x):
            if (not s._resolved):
                s.handleResolve(x)
        self.then(_hx_local_0)
        return s

    def handleResolve(self,val):
        if ((not self._end) and (not self._pause)):
            self._resolve(val)

    def pause(self,_hx_set = None):
        if (_hx_set is None):
            _hx_set = (not self._pause)
        self._pause = _hx_set

    def pipe(self,f):
        ret = promhx_Stream(None)
        ret1 = ret
        f1 = f
        linked = False
        def _hx_local_1(x):
            nonlocal linked
            if (not linked):
                linked = True
                pipe_ret = f1(x)
                _this = pipe_ret._update
                _this.append(_hx_AnonObject({'_hx_async': ret1, 'linkf': ret1.handleResolve}))
                def _hx_local_0(x):
                    return x
                promhx_base_AsyncBase.immediateLinkUpdate(pipe_ret,ret1,_hx_local_0)
        linkf = _hx_local_1
        _this = self._update
        _this.append(_hx_AnonObject({'_hx_async': ret1, 'linkf': linkf}))
        if (self._resolved and (not self._pending)):
            try:
                linkf(self._val)
            except BaseException as _g:
                None
                e = haxe_Exception.caught(_g).unwrap()
                ret1.handleError(e)
        def _hx_local_2(x):
            ret.end()
        self._end_promise.then(_hx_local_2)
        return ret

    def errorPipe(self,f):
        ret = promhx_Stream(None)
        def _hx_local_0(e):
            piped = f(e)
            piped.then(ret._resolve)
            piped._end_promise.then(ret._end_promise._resolve)
        self.catchError(_hx_local_0)
        self.then(ret._resolve)
        def _hx_local_1(x):
            ret.end()
        self._end_promise.then(_hx_local_1)
        return ret

    def handleEnd(self):
        if self._pending:
            promhx_base_EventLoop.queue.add(self.handleEnd)
            promhx_base_EventLoop.continueOnNextLoop()
        elif self._end_promise._resolved:
            return
        else:
            self._end = True
            o = (haxe_ds_Option.Some(self._val) if (self._resolved) else haxe_ds_Option._hx_None)
            self._end_promise.handleResolve(o)
            self._update = []
            self._error = []

    def end(self):
        promhx_base_EventLoop.queue.add(self.handleEnd)
        promhx_base_EventLoop.continueOnNextLoop()
        return self

    def endThen(self,f):
        return self._end_promise.then(f)

    def filter(self,f):
        ret = promhx_Stream(None)
        _this = self._update
        def _hx_local_0(x):
            if f(x):
                ret.handleResolve(x)
        _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': _hx_local_0}))
        def _hx_local_1(x):
            return x
        promhx_base_AsyncBase.immediateLinkUpdate(self,ret,_hx_local_1)
        return ret

    def concat(self,s):
        ret = promhx_Stream(None)
        _this = self._update
        _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': ret.handleResolve}))
        def _hx_local_0(x):
            return x
        promhx_base_AsyncBase.immediateLinkUpdate(self,ret,_hx_local_0)
        def _hx_local_3(_):
            def _hx_local_1(x):
                ret.handleResolve(x)
                return ret
            s.pipe(_hx_local_1)
            def _hx_local_2(_):
                ret.end()
            s._end_promise.then(_hx_local_2)
        self._end_promise.then(_hx_local_3)
        return ret

    def merge(self,s):
        ret = promhx_Stream(None)
        _this = self._update
        _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': ret.handleResolve}))
        _this = s._update
        _this.append(_hx_AnonObject({'_hx_async': ret, 'linkf': ret.handleResolve}))
        def _hx_local_0(x):
            return x
        promhx_base_AsyncBase.immediateLinkUpdate(self,ret,_hx_local_0)
        def _hx_local_1(x):
            return x
        promhx_base_AsyncBase.immediateLinkUpdate(s,ret,_hx_local_1)
        return ret

    @staticmethod
    def foreach(itb):
        s = promhx_Stream(None)
        i = HxOverrides.iterator(itb)
        while i.hasNext():
            i1 = i.next()
            s.handleResolve(i1)
        s.end()
        return s

    @staticmethod
    def wheneverAll(itb):
        ret = promhx_Stream(None)
        all = itb
        next = ret
        def _hx_local_0(arr,current,v):
            if ((len(arr) == 0) or promhx_base_AsyncBase.allFulfilled(arr)):
                _g = []
                a = HxOverrides.iterator(all)
                while a.hasNext():
                    a1 = a.next()
                    x = (v if ((a1 == current)) else a1._val)
                    _g.append(x)
                vals = _g
                next.handleResolve(vals)
        cthen = _hx_local_0
        a = HxOverrides.iterator(all)
        while a.hasNext():
            a1 = a.next()
            _this = a1._update
            next1 = next
            _g = [cthen]
            _g1 = []
            a2 = HxOverrides.iterator(all)
            while a2.hasNext():
                a21 = a2.next()
                if (a21 != a1):
                    _g1.append(a21)
            def _hx_local_2(current,arr,_g):
                def _hx_local_1(v):
                    (_g[0] if 0 < len(_g) else None)((arr[0] if 0 < len(arr) else None),(current[0] if 0 < len(current) else None),v)
                return _hx_local_1
            x = _hx_AnonObject({'_hx_async': next1, 'linkf': _hx_local_2([a1],[_g1],_g)})
            _this.append(x)
        if promhx_base_AsyncBase.allFulfilled(all):
            next1 = next
            _g1 = []
            a = HxOverrides.iterator(all)
            while a.hasNext():
                a1 = a.next()
                x = a1._val
                _g1.append(x)
            next1.handleResolve(_g1)
        return ret

    @staticmethod
    def concatAll(itb):
        ret = promhx_Stream(None)
        i = HxOverrides.iterator(itb)
        while i.hasNext():
            i1 = i.next()
            ret.concat(i1)
        return ret

    @staticmethod
    def mergeAll(itb):
        ret = promhx_Stream(None)
        i = HxOverrides.iterator(itb)
        while i.hasNext():
            i1 = i.next()
            ret.merge(i1)
        return ret

    @staticmethod
    def stream(_val):
        ret = promhx_Stream(None)
        ret.handleResolve(_val)
        return ret

promhx_Stream._hx_class = promhx_Stream


class promhx_PublicStream(promhx_Stream):
    _hx_class_name = "promhx.PublicStream"
    __slots__ = ()
    _hx_fields = []
    _hx_methods = ["resolve", "throwError", "update"]
    _hx_statics = ["publicstream"]
    _hx_interfaces = []
    _hx_super = promhx_Stream


    def __init__(self,_hx_def = None):
        super().__init__(_hx_def)

    def resolve(self,val):
        self.handleResolve(val)

    def throwError(self,e):
        self.handleError(e)

    def update(self,val):
        self.handleResolve(val)

    @staticmethod
    def publicstream(val):
        ps = promhx_PublicStream(None)
        ps.handleResolve(val)
        return ps

promhx_PublicStream._hx_class = promhx_PublicStream


class promhx_base_EventLoop:
    _hx_class_name = "promhx.base.EventLoop"
    __slots__ = ()
    _hx_statics = ["queue", "nextLoop", "enqueue", "set_nextLoop", "queueEmpty", "finish", "clear", "f", "continueOnNextLoop"]
    nextLoop = None

    @staticmethod
    def enqueue(eqf):
        promhx_base_EventLoop.queue.add(eqf)
        promhx_base_EventLoop.continueOnNextLoop()

    @staticmethod
    def set_nextLoop(f):
        if (promhx_base_EventLoop.nextLoop is not None):
            raise haxe_Exception.thrown("nextLoop has already been set")
        else:
            promhx_base_EventLoop.nextLoop = f
        return promhx_base_EventLoop.nextLoop

    @staticmethod
    def queueEmpty():
        return promhx_base_EventLoop.queue.isEmpty()

    @staticmethod
    def finish(max_iterations = None):
        if (max_iterations is None):
            max_iterations = 1000
        fn = None
        while True:
            tmp = None
            tmp1 = max_iterations
            max_iterations = (max_iterations - 1)
            if (tmp1 > 0):
                fn = promhx_base_EventLoop.queue.pop()
                tmp = (fn is not None)
            else:
                tmp = False
            if (not tmp):
                break
            fn()
        return promhx_base_EventLoop.queue.isEmpty()

    @staticmethod
    def clear():
        promhx_base_EventLoop.queue = haxe_ds_List()

    @staticmethod
    def f():
        fn = promhx_base_EventLoop.queue.pop()
        if (fn is not None):
            fn()
        if (not promhx_base_EventLoop.queue.isEmpty()):
            promhx_base_EventLoop.continueOnNextLoop()

    @staticmethod
    def continueOnNextLoop():
        if (promhx_base_EventLoop.nextLoop is not None):
            promhx_base_EventLoop.nextLoop(promhx_base_EventLoop.f)
        else:
            promhx_base_EventLoop.f()
promhx_base_EventLoop._hx_class = promhx_base_EventLoop

class promhx_error_PromiseError(Enum):
    __slots__ = ()
    _hx_class_name = "promhx.error.PromiseError"
    _hx_constructs = ["AlreadyResolved", "DownstreamNotFullfilled"]

    @staticmethod
    def AlreadyResolved(message):
        return promhx_error_PromiseError("AlreadyResolved", 0, (message,))

    @staticmethod
    def DownstreamNotFullfilled(message):
        return promhx_error_PromiseError("DownstreamNotFullfilled", 1, (message,))
promhx_error_PromiseError._hx_class = promhx_error_PromiseError


class python_Boot:
    _hx_class_name = "python.Boot"
    __slots__ = ()
    _hx_statics = ["keywords", "toString1", "fields", "simpleField", "hasField", "field", "getInstanceFields", "getSuperClass", "getClassFields", "prefixLength", "unhandleKeywords"]

    @staticmethod
    def toString1(o,s):
        if (o is None):
            return "null"
        if isinstance(o,str):
            return o
        if (s is None):
            s = ""
        if (len(s) >= 5):
            return "<...>"
        if isinstance(o,bool):
            if o:
                return "true"
            else:
                return "false"
        if (isinstance(o,int) and (not isinstance(o,bool))):
            return str(o)
        if isinstance(o,float):
            try:
                if (o == int(o)):
                    return str(Math.floor((o + 0.5)))
                else:
                    return str(o)
            except BaseException as _g:
                None
                return str(o)
        if isinstance(o,list):
            o1 = o
            l = len(o1)
            st = "["
            s = (("null" if s is None else s) + "\t")
            _g = 0
            _g1 = l
            while (_g < _g1):
                i = _g
                _g = (_g + 1)
                prefix = ""
                if (i > 0):
                    prefix = ","
                st = (("null" if st is None else st) + HxOverrides.stringOrNull(((("null" if prefix is None else prefix) + HxOverrides.stringOrNull(python_Boot.toString1((o1[i] if i >= 0 and i < len(o1) else None),s))))))
            st = (("null" if st is None else st) + "]")
            return st
        try:
            if hasattr(o,"toString"):
                return o.toString()
        except BaseException as _g:
            None
        if hasattr(o,"__class__"):
            if isinstance(o,_hx_AnonObject):
                toStr = None
                try:
                    fields = python_Boot.fields(o)
                    _g = []
                    _g1 = 0
                    while (_g1 < len(fields)):
                        f = (fields[_g1] if _g1 >= 0 and _g1 < len(fields) else None)
                        _g1 = (_g1 + 1)
                        x = ((("" + ("null" if f is None else f)) + " : ") + HxOverrides.stringOrNull(python_Boot.toString1(python_Boot.simpleField(o,f),(("null" if s is None else s) + "\t"))))
                        _g.append(x)
                    fieldsStr = _g
                    toStr = (("{ " + HxOverrides.stringOrNull(", ".join([x1 for x1 in fieldsStr]))) + " }")
                except BaseException as _g:
                    None
                    return "{ ... }"
                if (toStr is None):
                    return "{ ... }"
                else:
                    return toStr
            if isinstance(o,Enum):
                o1 = o
                l = len(o1.params)
                hasParams = (l > 0)
                if hasParams:
                    paramsStr = ""
                    _g = 0
                    _g1 = l
                    while (_g < _g1):
                        i = _g
                        _g = (_g + 1)
                        prefix = ""
                        if (i > 0):
                            prefix = ","
                        paramsStr = (("null" if paramsStr is None else paramsStr) + HxOverrides.stringOrNull(((("null" if prefix is None else prefix) + HxOverrides.stringOrNull(python_Boot.toString1(o1.params[i],s))))))
                    return (((HxOverrides.stringOrNull(o1.tag) + "(") + ("null" if paramsStr is None else paramsStr)) + ")")
                else:
                    return o1.tag
            if hasattr(o,"_hx_class_name"):
                if (o.__class__.__name__ != "type"):
                    fields = python_Boot.getInstanceFields(o)
                    _g = []
                    _g1 = 0
                    while (_g1 < len(fields)):
                        f = (fields[_g1] if _g1 >= 0 and _g1 < len(fields) else None)
                        _g1 = (_g1 + 1)
                        x = ((("" + ("null" if f is None else f)) + " : ") + HxOverrides.stringOrNull(python_Boot.toString1(python_Boot.simpleField(o,f),(("null" if s is None else s) + "\t"))))
                        _g.append(x)
                    fieldsStr = _g
                    toStr = (((HxOverrides.stringOrNull(o._hx_class_name) + "( ") + HxOverrides.stringOrNull(", ".join([x1 for x1 in fieldsStr]))) + " )")
                    return toStr
                else:
                    fields = python_Boot.getClassFields(o)
                    _g = []
                    _g1 = 0
                    while (_g1 < len(fields)):
                        f = (fields[_g1] if _g1 >= 0 and _g1 < len(fields) else None)
                        _g1 = (_g1 + 1)
                        x = ((("" + ("null" if f is None else f)) + " : ") + HxOverrides.stringOrNull(python_Boot.toString1(python_Boot.simpleField(o,f),(("null" if s is None else s) + "\t"))))
                        _g.append(x)
                    fieldsStr = _g
                    toStr = (((("#" + HxOverrides.stringOrNull(o._hx_class_name)) + "( ") + HxOverrides.stringOrNull(", ".join([x1 for x1 in fieldsStr]))) + " )")
                    return toStr
            if ((type(o) == type) and (o == str)):
                return "#String"
            if ((type(o) == type) and (o == list)):
                return "#Array"
            if callable(o):
                return "function"
            try:
                if hasattr(o,"__repr__"):
                    return o.__repr__()
            except BaseException as _g:
                None
            if hasattr(o,"__str__"):
                return o.__str__([])
            if hasattr(o,"__name__"):
                return o.__name__
            return "???"
        else:
            return str(o)

    @staticmethod
    def fields(o):
        a = []
        if (o is not None):
            if hasattr(o,"_hx_fields"):
                fields = o._hx_fields
                if (fields is not None):
                    return list(fields)
            if isinstance(o,_hx_AnonObject):
                d = o.__dict__
                keys = d.keys()
                handler = python_Boot.unhandleKeywords
                for k in keys:
                    if (k != '_hx_disable_getattr'):
                        a.append(handler(k))
            elif hasattr(o,"__dict__"):
                d = o.__dict__
                keys1 = d.keys()
                for k in keys1:
                    a.append(k)
        return a

    @staticmethod
    def simpleField(o,field):
        if (field is None):
            return None
        field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
        if hasattr(o,field1):
            return getattr(o,field1)
        else:
            return None

    @staticmethod
    def hasField(o,field):
        if isinstance(o,_hx_AnonObject):
            return o._hx_hasattr(field)
        return hasattr(o,(("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field)))

    @staticmethod
    def field(o,field):
        if (field is None):
            return None
        if isinstance(o,str):
            field1 = field
            _hx_local_0 = len(field1)
            if (_hx_local_0 == 10):
                if (field1 == "charCodeAt"):
                    return python_internal_MethodClosure(o,HxString.charCodeAt)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 11):
                if (field1 == "lastIndexOf"):
                    return python_internal_MethodClosure(o,HxString.lastIndexOf)
                elif (field1 == "toLowerCase"):
                    return python_internal_MethodClosure(o,HxString.toLowerCase)
                elif (field1 == "toUpperCase"):
                    return python_internal_MethodClosure(o,HxString.toUpperCase)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 9):
                if (field1 == "substring"):
                    return python_internal_MethodClosure(o,HxString.substring)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 5):
                if (field1 == "split"):
                    return python_internal_MethodClosure(o,HxString.split)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 7):
                if (field1 == "indexOf"):
                    return python_internal_MethodClosure(o,HxString.indexOf)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 8):
                if (field1 == "toString"):
                    return python_internal_MethodClosure(o,HxString.toString)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_0 == 6):
                if (field1 == "charAt"):
                    return python_internal_MethodClosure(o,HxString.charAt)
                elif (field1 == "length"):
                    return len(o)
                elif (field1 == "substr"):
                    return python_internal_MethodClosure(o,HxString.substr)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            else:
                field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                if hasattr(o,field1):
                    return getattr(o,field1)
                else:
                    return None
        elif isinstance(o,list):
            field1 = field
            _hx_local_1 = len(field1)
            if (_hx_local_1 == 11):
                if (field1 == "lastIndexOf"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.lastIndexOf)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 4):
                if (field1 == "copy"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.copy)
                elif (field1 == "join"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.join)
                elif (field1 == "push"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.push)
                elif (field1 == "sort"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.sort)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 5):
                if (field1 == "shift"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.shift)
                elif (field1 == "slice"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.slice)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 7):
                if (field1 == "indexOf"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.indexOf)
                elif (field1 == "reverse"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.reverse)
                elif (field1 == "unshift"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.unshift)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 3):
                if (field1 == "map"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.map)
                elif (field1 == "pop"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.pop)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 8):
                if (field1 == "contains"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.contains)
                elif (field1 == "iterator"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.iterator)
                elif (field1 == "toString"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.toString)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 16):
                if (field1 == "keyValueIterator"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.keyValueIterator)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            elif (_hx_local_1 == 6):
                if (field1 == "concat"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.concat)
                elif (field1 == "filter"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.filter)
                elif (field1 == "insert"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.insert)
                elif (field1 == "length"):
                    return len(o)
                elif (field1 == "remove"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.remove)
                elif (field1 == "splice"):
                    return python_internal_MethodClosure(o,python_internal_ArrayImpl.splice)
                else:
                    field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                    if hasattr(o,field1):
                        return getattr(o,field1)
                    else:
                        return None
            else:
                field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
                if hasattr(o,field1):
                    return getattr(o,field1)
                else:
                    return None
        else:
            field1 = (("_hx_" + field) if ((field in python_Boot.keywords)) else (("_hx_" + field) if (((((len(field) > 2) and ((ord(field[0]) == 95))) and ((ord(field[1]) == 95))) and ((ord(field[(len(field) - 1)]) != 95)))) else field))
            if hasattr(o,field1):
                return getattr(o,field1)
            else:
                return None

    @staticmethod
    def getInstanceFields(c):
        f = (list(c._hx_fields) if (hasattr(c,"_hx_fields")) else [])
        if hasattr(c,"_hx_methods"):
            f = (f + c._hx_methods)
        sc = python_Boot.getSuperClass(c)
        if (sc is None):
            return f
        else:
            scArr = python_Boot.getInstanceFields(sc)
            scMap = set(scArr)
            _g = 0
            while (_g < len(f)):
                f1 = (f[_g] if _g >= 0 and _g < len(f) else None)
                _g = (_g + 1)
                if (not (f1 in scMap)):
                    scArr.append(f1)
            return scArr

    @staticmethod
    def getSuperClass(c):
        if (c is None):
            return None
        try:
            if hasattr(c,"_hx_super"):
                return c._hx_super
            return None
        except BaseException as _g:
            None
        return None

    @staticmethod
    def getClassFields(c):
        if hasattr(c,"_hx_statics"):
            x = c._hx_statics
            return list(x)
        else:
            return []

    @staticmethod
    def unhandleKeywords(name):
        if (HxString.substr(name,0,python_Boot.prefixLength) == "_hx_"):
            real = HxString.substr(name,python_Boot.prefixLength,None)
            if (real in python_Boot.keywords):
                return real
        return name
python_Boot._hx_class = python_Boot


class python_HaxeIterator:
    _hx_class_name = "python.HaxeIterator"
    __slots__ = ("it", "x", "has", "checked")
    _hx_fields = ["it", "x", "has", "checked"]
    _hx_methods = ["next", "hasNext"]

    def __init__(self,it):
        self.checked = False
        self.has = False
        self.x = None
        self.it = it

    def next(self):
        if (not self.checked):
            self.hasNext()
        self.checked = False
        return self.x

    def hasNext(self):
        if (not self.checked):
            try:
                self.x = self.it.__next__()
                self.has = True
            except BaseException as _g:
                None
                if Std.isOfType(haxe_Exception.caught(_g).unwrap(),StopIteration):
                    self.has = False
                    self.x = None
                else:
                    raise _g
            self.checked = True
        return self.has

python_HaxeIterator._hx_class = python_HaxeIterator


class python__KwArgs_KwArgs_Impl_:
    _hx_class_name = "python._KwArgs.KwArgs_Impl_"
    __slots__ = ()
    _hx_statics = ["fromT"]

    @staticmethod
    def fromT(d):
        this1 = python_Lib.anonAsDict(d)
        return this1
python__KwArgs_KwArgs_Impl_._hx_class = python__KwArgs_KwArgs_Impl_


class python_Lib:
    _hx_class_name = "python.Lib"
    __slots__ = ()
    _hx_statics = ["lineEnd", "printString", "anonToDict", "anonAsDict"]

    @staticmethod
    def printString(_hx_str):
        encoding = "utf-8"
        if (encoding is None):
            encoding = "utf-8"
        python_lib_Sys.stdout.buffer.write(_hx_str.encode(encoding, "strict"))
        python_lib_Sys.stdout.flush()

    @staticmethod
    def anonToDict(o):
        if isinstance(o,_hx_AnonObject):
            return o.__dict__.copy()
        else:
            return None

    @staticmethod
    def anonAsDict(o):
        if isinstance(o,_hx_AnonObject):
            return o.__dict__
        else:
            return None
python_Lib._hx_class = python_Lib


class python_internal_ArrayImpl:
    _hx_class_name = "python.internal.ArrayImpl"
    __slots__ = ()
    _hx_statics = ["concat", "copy", "iterator", "keyValueIterator", "indexOf", "lastIndexOf", "join", "toString", "pop", "push", "unshift", "remove", "contains", "shift", "slice", "sort", "splice", "map", "filter", "insert", "reverse", "_get", "_set"]

    @staticmethod
    def concat(a1,a2):
        return (a1 + a2)

    @staticmethod
    def copy(x):
        return list(x)

    @staticmethod
    def iterator(x):
        return python_HaxeIterator(x.__iter__())

    @staticmethod
    def keyValueIterator(x):
        return haxe_iterators_ArrayKeyValueIterator(x)

    @staticmethod
    def indexOf(a,x,fromIndex = None):
        _hx_len = len(a)
        l = (0 if ((fromIndex is None)) else ((_hx_len + fromIndex) if ((fromIndex < 0)) else fromIndex))
        if (l < 0):
            l = 0
        _g = l
        _g1 = _hx_len
        while (_g < _g1):
            i = _g
            _g = (_g + 1)
            if HxOverrides.eq(a[i],x):
                return i
        return -1

    @staticmethod
    def lastIndexOf(a,x,fromIndex = None):
        _hx_len = len(a)
        l = (_hx_len if ((fromIndex is None)) else (((_hx_len + fromIndex) + 1) if ((fromIndex < 0)) else (fromIndex + 1)))
        if (l > _hx_len):
            l = _hx_len
        while True:
            l = (l - 1)
            tmp = l
            if (not ((tmp > -1))):
                break
            if HxOverrides.eq(a[l],x):
                return l
        return -1

    @staticmethod
    def join(x,sep):
        return sep.join([python_Boot.toString1(x1,'') for x1 in x])

    @staticmethod
    def toString(x):
        return (("[" + HxOverrides.stringOrNull(",".join([python_Boot.toString1(x1,'') for x1 in x]))) + "]")

    @staticmethod
    def pop(x):
        if (len(x) == 0):
            return None
        else:
            return x.pop()

    @staticmethod
    def push(x,e):
        x.append(e)
        return len(x)

    @staticmethod
    def unshift(x,e):
        x.insert(0, e)

    @staticmethod
    def remove(x,e):
        try:
            x.remove(e)
            return True
        except BaseException as _g:
            None
            return False

    @staticmethod
    def contains(x,e):
        return (e in x)

    @staticmethod
    def shift(x):
        if (len(x) == 0):
            return None
        return x.pop(0)

    @staticmethod
    def slice(x,pos,end = None):
        return x[pos:end]

    @staticmethod
    def sort(x,f):
        x.sort(key= python_lib_Functools.cmp_to_key(f))

    @staticmethod
    def splice(x,pos,_hx_len):
        if (pos < 0):
            pos = (len(x) + pos)
        if (pos < 0):
            pos = 0
        res = x[pos:(pos + _hx_len)]
        del x[pos:(pos + _hx_len)]
        return res

    @staticmethod
    def map(x,f):
        return list(map(f,x))

    @staticmethod
    def filter(x,f):
        return list(filter(f,x))

    @staticmethod
    def insert(a,pos,x):
        a.insert(pos, x)

    @staticmethod
    def reverse(a):
        a.reverse()

    @staticmethod
    def _get(x,idx):
        if ((idx > -1) and ((idx < len(x)))):
            return x[idx]
        else:
            return None

    @staticmethod
    def _set(x,idx,v):
        l = len(x)
        while (l < idx):
            x.append(None)
            l = (l + 1)
        if (l == idx):
            x.append(v)
        else:
            x[idx] = v
        return v
python_internal_ArrayImpl._hx_class = python_internal_ArrayImpl


class HxOverrides:
    _hx_class_name = "HxOverrides"
    __slots__ = ()
    _hx_statics = ["iterator", "eq", "stringOrNull", "mapKwArgs"]

    @staticmethod
    def iterator(x):
        if isinstance(x,list):
            return haxe_iterators_ArrayIterator(x)
        return x.iterator()

    @staticmethod
    def eq(a,b):
        if (isinstance(a,list) or isinstance(b,list)):
            return a is b
        return (a == b)

    @staticmethod
    def stringOrNull(s):
        if (s is None):
            return "null"
        else:
            return s

    @staticmethod
    def mapKwArgs(a,v):
        a1 = _hx_AnonObject(python_Lib.anonToDict(a))
        k = python_HaxeIterator(iter(v.keys()))
        while k.hasNext():
            k1 = k.next()
            val = v.get(k1)
            if a1._hx_hasattr(k1):
                x = getattr(a1,k1)
                setattr(a1,val,x)
                delattr(a1,k1)
        return a1
HxOverrides._hx_class = HxOverrides


class python_internal_MethodClosure:
    _hx_class_name = "python.internal.MethodClosure"
    __slots__ = ("obj", "func")
    _hx_fields = ["obj", "func"]
    _hx_methods = ["__call__"]

    def __init__(self,obj,func):
        self.obj = obj
        self.func = func

    def __call__(self,*args):
        return self.func(self.obj,*args)

python_internal_MethodClosure._hx_class = python_internal_MethodClosure


class HxString:
    _hx_class_name = "HxString"
    __slots__ = ()
    _hx_statics = ["split", "charCodeAt", "charAt", "lastIndexOf", "toUpperCase", "toLowerCase", "indexOf", "indexOfImpl", "toString", "substring", "substr"]

    @staticmethod
    def split(s,d):
        if (d == ""):
            return list(s)
        else:
            return s.split(d)

    @staticmethod
    def charCodeAt(s,index):
        if ((((s is None) or ((len(s) == 0))) or ((index < 0))) or ((index >= len(s)))):
            return None
        else:
            return ord(s[index])

    @staticmethod
    def charAt(s,index):
        if ((index < 0) or ((index >= len(s)))):
            return ""
        else:
            return s[index]

    @staticmethod
    def lastIndexOf(s,_hx_str,startIndex = None):
        if (startIndex is None):
            return s.rfind(_hx_str, 0, len(s))
        elif (_hx_str == ""):
            length = len(s)
            if (startIndex < 0):
                startIndex = (length + startIndex)
                if (startIndex < 0):
                    startIndex = 0
            if (startIndex > length):
                return length
            else:
                return startIndex
        else:
            i = s.rfind(_hx_str, 0, (startIndex + 1))
            startLeft = (max(0,((startIndex + 1) - len(_hx_str))) if ((i == -1)) else (i + 1))
            check = s.find(_hx_str, startLeft, len(s))
            if ((check > i) and ((check <= startIndex))):
                return check
            else:
                return i

    @staticmethod
    def toUpperCase(s):
        return s.upper()

    @staticmethod
    def toLowerCase(s):
        return s.lower()

    @staticmethod
    def indexOf(s,_hx_str,startIndex = None):
        if (startIndex is None):
            return s.find(_hx_str)
        else:
            return HxString.indexOfImpl(s,_hx_str,startIndex)

    @staticmethod
    def indexOfImpl(s,_hx_str,startIndex):
        if (_hx_str == ""):
            length = len(s)
            if (startIndex < 0):
                startIndex = (length + startIndex)
                if (startIndex < 0):
                    startIndex = 0
            if (startIndex > length):
                return length
            else:
                return startIndex
        return s.find(_hx_str, startIndex)

    @staticmethod
    def toString(s):
        return s

    @staticmethod
    def substring(s,startIndex,endIndex = None):
        if (startIndex < 0):
            startIndex = 0
        if (endIndex is None):
            return s[startIndex:]
        else:
            if (endIndex < 0):
                endIndex = 0
            if (endIndex < startIndex):
                return s[endIndex:startIndex]
            else:
                return s[startIndex:endIndex]

    @staticmethod
    def substr(s,startIndex,_hx_len = None):
        if (_hx_len is None):
            return s[startIndex:]
        else:
            if (_hx_len == 0):
                return ""
            if (startIndex < 0):
                startIndex = (len(s) + startIndex)
                if (startIndex < 0):
                    startIndex = 0
            return s[startIndex:(startIndex + _hx_len)]
HxString._hx_class = HxString


class sys_thread_EventLoop:
    _hx_class_name = "sys.thread.EventLoop"
    __slots__ = ("mutex", "oneTimeEvents", "oneTimeEventsIdx", "waitLock", "promisedEventsCount", "regularEvents")
    _hx_fields = ["mutex", "oneTimeEvents", "oneTimeEventsIdx", "waitLock", "promisedEventsCount", "regularEvents"]
    _hx_methods = ["loop"]

    def __init__(self):
        self.regularEvents = None
        self.promisedEventsCount = 0
        self.waitLock = sys_thread_Lock()
        self.oneTimeEventsIdx = 0
        self.oneTimeEvents = list()
        self.mutex = sys_thread_Mutex()

    def loop(self):
        recycleRegular = []
        recycleOneTimers = []
        while True:
            now = python_lib_Time.time()
            regularsToRun = recycleRegular
            eventsToRunIdx = 0
            nextEventAt = -1
            self.mutex.lock.acquire(True)
            while self.waitLock.semaphore.acquire(True,0.0):
                pass
            current = self.regularEvents
            while (current is not None):
                if (current.nextRunTime <= now):
                    tmp = eventsToRunIdx
                    eventsToRunIdx = (eventsToRunIdx + 1)
                    python_internal_ArrayImpl._set(regularsToRun, tmp, current)
                    current.nextRunTime = (current.nextRunTime + current.interval)
                    nextEventAt = -2
                elif ((nextEventAt == -1) or ((current.nextRunTime < nextEventAt))):
                    nextEventAt = current.nextRunTime
                current = current.next
            self.mutex.lock.release()
            _g = 0
            _g1 = eventsToRunIdx
            while (_g < _g1):
                i = _g
                _g = (_g + 1)
                if (not (regularsToRun[i] if i >= 0 and i < len(regularsToRun) else None).cancelled):
                    (regularsToRun[i] if i >= 0 and i < len(regularsToRun) else None).run()
                python_internal_ArrayImpl._set(regularsToRun, i, None)
            eventsToRunIdx = 0
            oneTimersToRun = recycleOneTimers
            self.mutex.lock.acquire(True)
            _g2_current = 0
            _g2_array = self.oneTimeEvents
            while (_g2_current < len(_g2_array)):
                _g3_value = (_g2_array[_g2_current] if _g2_current >= 0 and _g2_current < len(_g2_array) else None)
                _g3_key = _g2_current
                _g2_current = (_g2_current + 1)
                i1 = _g3_key
                event = _g3_value
                if (event is None):
                    break
                else:
                    tmp1 = eventsToRunIdx
                    eventsToRunIdx = (eventsToRunIdx + 1)
                    python_internal_ArrayImpl._set(oneTimersToRun, tmp1, event)
                    python_internal_ArrayImpl._set(self.oneTimeEvents, i1, None)
            self.oneTimeEventsIdx = 0
            hasPromisedEvents = (self.promisedEventsCount > 0)
            self.mutex.lock.release()
            _g2 = 0
            _g3 = eventsToRunIdx
            while (_g2 < _g3):
                i2 = _g2
                _g2 = (_g2 + 1)
                (oneTimersToRun[i2] if i2 >= 0 and i2 < len(oneTimersToRun) else None)()
                python_internal_ArrayImpl._set(oneTimersToRun, i2, None)
            if (eventsToRunIdx > 0):
                nextEventAt = -2
            r_nextEventAt = nextEventAt
            r_anyTime = hasPromisedEvents
            _g4 = r_anyTime
            _g5 = r_nextEventAt
            _g6 = _g5
            if (_g6 == -2):
                pass
            elif (_g6 == -1):
                if _g4:
                    self.waitLock.semaphore.acquire(True,None)
                else:
                    break
            else:
                time = _g5
                timeout = (time - python_lib_Time.time())
                _this = self.waitLock
                timeout1 = (0 if (python_lib_Math.isnan(0)) else (timeout if (python_lib_Math.isnan(timeout)) else max(0,timeout)))
                _this.semaphore.acquire(True,timeout1)

sys_thread_EventLoop._hx_class = sys_thread_EventLoop


class sys_thread__EventLoop_RegularEvent:
    _hx_class_name = "sys.thread._EventLoop.RegularEvent"
    __slots__ = ("nextRunTime", "interval", "run", "next", "cancelled")
    _hx_fields = ["nextRunTime", "interval", "run", "next", "cancelled"]

    def __init__(self,run,nextRunTime,interval):
        self.next = None
        self.cancelled = False
        self.run = run
        self.nextRunTime = nextRunTime
        self.interval = interval

sys_thread__EventLoop_RegularEvent._hx_class = sys_thread__EventLoop_RegularEvent


class sys_thread_Lock:
    _hx_class_name = "sys.thread.Lock"
    __slots__ = ("semaphore",)
    _hx_fields = ["semaphore"]

    def __init__(self):
        self.semaphore = Lock(0)

sys_thread_Lock._hx_class = sys_thread_Lock


class sys_thread_Mutex:
    _hx_class_name = "sys.thread.Mutex"
    __slots__ = ("lock",)
    _hx_fields = ["lock"]

    def __init__(self):
        self.lock = sys_thread__Mutex_NativeRLock()

sys_thread_Mutex._hx_class = sys_thread_Mutex


class sys_thread__Thread_Thread_Impl_:
    _hx_class_name = "sys.thread._Thread.Thread_Impl_"
    __slots__ = ()
    _hx_statics = ["processEvents"]

    @staticmethod
    def processEvents():
        sys_thread__Thread_HxThread.current().events.loop()
sys_thread__Thread_Thread_Impl_._hx_class = sys_thread__Thread_Thread_Impl_


class sys_thread__Thread_HxThread:
    _hx_class_name = "sys.thread._Thread.HxThread"
    __slots__ = ("events", "nativeThread")
    _hx_fields = ["events", "nativeThread"]
    _hx_statics = ["threads", "threadsMutex", "mainThread", "current"]

    def __init__(self,t):
        self.events = None
        self.nativeThread = t
    threads = None
    threadsMutex = None
    mainThread = None

    @staticmethod
    def current():
        sys_thread__Thread_HxThread.threadsMutex.lock.acquire(True)
        ct = threading.current_thread()
        if (ct == threading.main_thread()):
            sys_thread__Thread_HxThread.threadsMutex.lock.release()
            return sys_thread__Thread_HxThread.mainThread
        if (not (ct in sys_thread__Thread_HxThread.threads.h)):
            sys_thread__Thread_HxThread.threads.set(ct,sys_thread__Thread_HxThread(ct))
        t = sys_thread__Thread_HxThread.threads.h.get(ct,None)
        sys_thread__Thread_HxThread.threadsMutex.lock.release()
        return t

sys_thread__Thread_HxThread._hx_class = sys_thread__Thread_HxThread

Math.NEGATIVE_INFINITY = float("-inf")
Math.POSITIVE_INFINITY = float("inf")
Math.NaN = float("nan")
Math.PI = python_lib_Math.pi
sys_thread__Thread_HxThread.threads = haxe_ds_ObjectMap()
sys_thread__Thread_HxThread.threadsMutex = sys_thread_Mutex()
sys_thread__Thread_HxThread.mainThread = sys_thread__Thread_HxThread(threading.current_thread())
sys_thread__Thread_HxThread.mainThread.events = sys_thread_EventLoop()

buddy_BuddySuite.useDefaultTrace = False
buddy_reporting__TraceReporter_Color_Impl_.Default = 39
buddy_reporting__TraceReporter_Color_Impl_.Red = 31
buddy_reporting__TraceReporter_Color_Impl_.Yellow = 33
buddy_reporting__TraceReporter_Color_Impl_.Green = 32
buddy_reporting__TraceReporter_Color_Impl_.White = 37
buddy_tests_SelfTest.lastSpec = buddy_Spec("No spec","No filename")
buddy_tests_SelfTest.lastSuite = buddy_Suite("No suite")
glm_GLM.EPSILON = 0.0000001
headbutt_threed_Headbutt.MAX_ITERATIONS = 20
headbutt_twod_Headbutt.MAX_ITERATIONS = 20
headbutt_twod_Headbutt.MAX_INTERSECTION_ITERATIONS = 32
headbutt_twod_Headbutt.INTERSECTION_EPSILON = 0.00001
promhx_base_EventLoop.queue = haxe_ds_List()
python_Boot.keywords = set(["and", "del", "from", "not", "with", "as", "elif", "global", "or", "yield", "assert", "else", "if", "pass", "None", "break", "except", "import", "raise", "True", "class", "exec", "in", "return", "False", "continue", "finally", "is", "try", "def", "for", "lambda", "while"])
python_Boot.prefixLength = len("_hx_")
python_Lib.lineEnd = ("\r\n" if ((Sys.systemName() == "Windows")) else "\n")

# TestMain.main()
# sys_thread__Thread_Thread_Impl_.processEvents()
