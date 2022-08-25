# flake8: noqa
"""
class DEigerClient provides an interface to the EIGER API

Author: Volker Pilipp, mod SasG
Contact: support@dectris.com


Copyright (c) 2022 DECTRIS Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import base64
import os.path

import json
import re
import sys
import socket
import fnmatch
import shutil


"""
Bad but working python3 backwards compability
"""
try:
    import http.client as httplibClient
except ImportError:
    import httplib as httplibClient

try:
    import urllib.request as urllibRequest
except ImportError:
    import urllib2 as urllibRequest


Version = '1.8.0'


# noinspection PyInterpreter
class DEigerClient:
    """
    class DEigerClient provides a low level interface to the EIGER API
    """

    def __init__(self, host = '127.0.0.1', port = 80, verbose = False, urlPrefix = None, user = None):
        """
        Create a client object to talk to the EIGER API.
        Args:
            host: hostname of the detector computer
            port: port usually 80 (http)
            verbose: bool value
            urlPrefix: String prepended to the urls. Should be None. Added for future convenience.
            user: "username:password". Should be None. Added for future convenience.
        """
        super().__init__()
        self._host = host
        self._port = port
        self._version = Version
        self._verbose = verbose
        self._urlPrefix = ""
        self._user = None
        self._connectionTimeout = 24*3600
        self._connection = httplibClient.HTTPConnection(self._host,self._port, timeout = self._connectionTimeout)
        self._serializer = None

        self.setUrlPrefix(urlPrefix)
        self.setUser(user)

    def serializer(self):
        """
        The serializer object shall have the methods loads(string) and dumps(obj), which load
        the string from json into a python object or store a python object into a json string
        """
        return self._serializer

    def setSerializer(self,serializer):
        """
        Set an explicit serializer object that converts native python objects to json string and vice versa.
        The serializer object shall have the methods loads(string) and dumps(obj), which load
        the string from json into a python object or store a python object into a json string
        """
        self._serializer = serializer

    def setVerbose(self,verbose):
        """ Switch verbose mode on and off.
        Args:
            verbose: bool value
        """
        self._verbose = bool(verbose)

    def setConnectionTimeout(self, timeout):
        """
        If DEigerClient has not received an reply from EIGER after
        timeout seconds, the request is aborted. timeout should be at
        least as long as the triggering command takes.
        Args:
            timeout timeout in seconds
        """
        self._connectionTimeout = timeout
        self._connection = httplibClient.HTTPConnection(self._host,self._port, timeout = self._connectionTimeout)

    def setUrlPrefix(self, urlPrefix):
        """Set url prefix, which is the string that is prepended to the
        urls. There is usually no need to call the command explicitly.
        Args:
           urlPrefix: String
        """
        if urlPrefix is None:
            self._urlPrefix = ""
        else:
            self._urlPrefix = str(urlPrefix)
            if len(self._urlPrefix) > 0 and self._urlPrefix[-1] != "/":
                self._urlPrefix += "/"

    def setUser(self, user):
        """
        Set username and password for basic authentication.
        There is usually no need to call the command explicitly.
        Args:
           user: String of the form username:password
        """
        if user is None:
            self._user = None
        else:
            self._user = base64.encodestring(user).replace('\n', '')



    def version(self,module = 'detector'):
        """
        Get version of a api module (i.e. 'detector', 'filewriter')
        Args:
            module: 'detector' or 'filewriter'
        """
        return self._getRequest(url = f'/{self._urlPrefix}{module}/api/version/')

    def sendSystemCommand(self, command):
        """
        Sending command "restart" restarts the SIMPLON API on the EIGER control unit
        """
        return self._putRequest(self._url('system','command',command), dataType = 'native', data = None)

    def sendStreamCommand(self, command):
        """
        Sending command "initialize" restarts the stream interface and disables it
        """
        return self._putRequest(self._url('stream','command',command), dataType = 'native', data = None)

    def listDetectorConfigParams(self):
        """Get list of all detector configuration parameters (param arg of configuration() and setConfiguration()).
        Convenience function, that does detectorConfig(param = 'keys')
        Returns:
            List of parameters.
        """
        return self.detectorConfig('keys')

    def detectorConfig(self,param = None, dataType = None):
        """Get detector configuration parameter
        Args:
            param: query the configuration parameter param, if None get full configuration, if 'keys' get all configuration parameters.
            dataType: None (= 'native'), 'native' ( return native python object) or 'tif' (return tif data).
        Returns:
            If param is None get configuration, if param is 'keys' return list of all parameters, else return the value of
            the parameter. If dataType is 'native' a dictionary is returned that may contain the keys: value, min, max,
            allowed_values, unit, value_type and access_mode. If dataType is 'tif', tiff formated data is returned as a python
            string.
        """
        return self._getRequest(self._url('detector','config',param),dataType)

    def setDetectorConfig(self, param, value, dataType = None):
        """
        Set detector configuration parameter param.
        Args:
            param: Parameter
            value: Value to set. If dataType is 'tif' value may be a string containing the tiff data or
                   a file object pointing to a tiff file.
            dataType: None, 'native' or 'tif'. If None, the data type is auto determined. If 'native' value
                      may be a native python object (e.g. int, float, str), if 'tif' value shell contain a
                      tif file (python string or file object to tif file).
        Returns:
            List of changed parameters.
        """
        return self._putRequest(self._url('detector','config',param), dataType, value)

    def setDetectorConfigMultiple(self,*params):
        """
        Convenience function that calls setDetectorConfig(param,value,dataType = None) for
        every pair param, value in *params.
        Args:
            *params: List of successive params of the form param0, value0, param1, value1, ...
                     The parameters are set in the same order they appear in *params.
        Returns:
            List of changed parameters.
        """
        changeList = []
        p = None
        for x in params:
            if p is None:
                p = x
            else:
                data = x
                changeList += self.setDetectorConfig(param = p, value = data, dataType = None)
                p = None
        return list(set(changeList))

    def listDetectorCommands(self):
        """
        Get list of all commands that may be sent to EIGER via sendDetectorCommand().
        Returns:
            List of commands
        """
        return self._getRequest(self._url('detector','command','keys'))

    def sendDetectorCommand(self,  command, parameter = None):
        """
        Send command to EIGER. The list of all available commands is obtained via listCommands().
        Args:
            command: Detector command
            parameter: Call command with parameter. If command = "trigger" a float parameter may be passed
        Returns:
            The commands 'arm' and 'trigger' return a dictionary containing 'sequence id'.
        """
        return self._putRequest(self._url('detector','command',command), dataType = 'native', data = parameter)


    def detectorStatus(self, param = 'keys'):
        """Get detector status information
        Args:
            param: query the status parameter param, if 'keys' get all status parameters.
        Returns:
            If param is None get configuration, if param is 'keys' return list of all parameters, else return dictionary
            that may contain the keys: value, value_type, unit, time, state, critical_limits, critical_values
        """
        return self._getRequest(self._url('detector','status',parameter = param))


    def fileWriterConfig(self,param = 'keys'):
        """Get filewriter configuration parameter
        Args:
            param: query the configuration parameter param, if 'keys' get all configuration parameters.
        Returns:
            If param is None get configuration, if param is 'keys' return list of all parameters, else return dictionary
            that may contain the keys: value, min, max, allowed_values, unit, value_type and access_mode
        """
        return self._getRequest(self._url('filewriter','config',parameter = param))

    def setFileWriterConfig(self,param,value):
        """
        Set file writer configuration parameter param.
        Args:
            param: parameter
            value: value to set
        Returns:
            List of changed parameters.
        """
        return self._putRequest(self._url('filewriter','config',parameter = param), dataType = 'native', data = value)

    def sendFileWriterCommand(self, command):
        """
        Send filewriter command to EIGER.
        Args:
            command: Command to send (up to now only "clear")
        Returns:
            Empty string
        """
        return self._putRequest(self._url("filewriter","command",parameter = command), dataType = "native")


    def fileWriterStatus(self,param = 'keys'):
        """Get filewriter status information
        Args:
            param: query the status parameter param, if 'keys' get all status parameters.
        Returns:
            If param is None get configuration, if param is 'keys' return list of all parameters, else return dictionary
            that may contain the keys: value, value_type, unit, time, state, critical_limits, critical_values
        """
        return self._getRequest(self._url('filewriter','status',parameter = param))

    def fileWriterFiles(self, filename = None, method = 'GET'):
        """
        Obtain file from detector.
        Args:
             filename: Name of file on the detector side. If None return list of available files
             method: 'GET' (get the content of the file) or 'DELETE' (delete file from server)
        Returns:
            List of available files if 'filename' is None,
            else if method is 'GET' the content of the file.
        """
        if method == 'GET':
            if filename is None:
                return self._getRequest(self._url('filewriter','files'))
            else:
                return self._getRequest(url = f'/{self._urlPrefix}data/{filename}', dataType = 'hdf5')
        elif method == 'DELETE':
            return self._delRequest(url = f'/{self._urlPrefix}data/{filename}')
        else:
            raise RuntimeError(f'Unknown method {method}')

    def fileWriterSave(self,filename,targetDir,regex = False):
        """
        Saves filename in targetDir. If regex is True, filename is considered to be a regular expression.
        Save all files that match filename
        Args:
            filename: Name of source file, may contain the wildcards '*' and '?' or regular expressions
            targetDir: Directory, where to store the files
        """
        if regex:
            pattern = re.compile(filename)
            [ self.fileWriterSave(f,targetDir)  for f in self.fileWriterFiles() if pattern.match(f) ]
        elif any([ c in filename for c in ['*','?','[',']'] ] ):
            # for f in self.fileWriterFiles():
            #    self._log('DEBUG ', f, '  ', fnmatch.fnmatch(f,filename))
            [ self.fileWriterSave(f,targetDir)  for f in self.fileWriterFiles() if fnmatch.fnmatch(f,filename) ]
        else:
            targetPath = os.path.join(targetDir,filename)
            url = f'http://{self._host}:{self._port}/{self._urlPrefix}data/{filename}'
            req = urllibRequest.urlopen(url, timeout = self._connectionTimeout)
            with open(targetPath, 'wb') as fp:
                self._log('Writing ', targetPath)
                shutil.copyfileobj(req, fp, 512*1024)
            # self._getRequest(url = '/{0}data/{1}'.format(self._urlPrefix, filename), dataType = 'hdf5',fileId = targetFile)
            # targetFile.write(self.fileWriterFiles(filename))
            assert os.access(targetPath,os.R_OK)
        return

    def monitorConfig(self,param = 'keys'):
        """Get monitor configuration parameter
        Args:
            param: query the configuration parameter param, if 'keys' get all configuration parameters.
        Returns:
            If param is 'keys' return list of all parameters, else return dictionary
            that may contain the keys: value, min, max, allowed_values, unit, value_type and access_mode
        """
        return self._getRequest(self._url('monitor','config',parameter = param))

    def setMonitorConfig(self,param,value):
        """
        Set monitor configuration parameter param.
        Args:
            param: parameter
            value: value to set
        Returns:
            List of changed parameters.
        """
        return self._putRequest(self._url('monitor','config',parameter = param), dataType = 'native', data = value)

    def monitorImages(self, param = None):
        """
        Obtain file from detector.
        Args:
             param: Either None (return list of available frames) or "monitor" (return latest frame),
                    "next"  (next image from buffer) or tuple(sequence id, image id) (return specific image)
        Returns:
            List of available frames (param = None) or tiff content of image file (param = "next", "monitor", (seqId,imgId))
        """
        if param is None:
            return self._getRequest(self._url('monitor','images',parameter = None) )
        elif param == "next":
            return self._getRequest(self._url('monitor',"images", parameter = "next"), dataType = "tif")
        elif param == "monitor":
            return self._getRequest(self._url('monitor','images',parameter = "monitor"), dataType = "tif")
        else:
            try:
                seqId = int(param[0])
                imgId = int(param[1])
                return self._getRequest(self._url('monitor',"images", parameter = f"{seqId}/{imgId}" ), dataType = 'tif')
            except (TypeError, ValueError):
                pass
        raise RuntimeError(f'Invalid parameter {param}')

    def monitorSave(self, param, path):
        """
        Save frame to path as tiff file.
        Args:
            param: Either None (return list of available frames) or "monitor" (return latest frame),
                   "next"  (next image from buffer) or tuple(sequence id, image id) (return specific image)
        Returns:
            None
        """
        data = None
        if param in ["next","monitor"]:
            data = self.monitorImages(param)
        else :
            try:
                int(param[0])
                int(param[1])
                data = self.monitorImages(param)
            except (TypeError, ValueError):
                pass
        if data is None:
            raise RuntimeError(f'Invalid parameter {param}')
        else:
            with open(path,'wb') as f:
                self._log('Writing ', path)
                f.write(data)
            assert os.access(path,os.R_OK)
        return

    def monitorStatus(self, param = "keys"):
        """
        Get monitor status information
        Args:
            param: query the status parameter param, if 'keys' get all status parameters.
        Returns:
            Dictionary that may contain the keys: value, value_type, unit, time, state,
            critical_limits, critical_values
        """
        return self._getRequest(self._url('monitor','status',parameter = param))

    def sendMonitorCommand(self, command):
        """
        Send monitor command to EIGER.
        Args:
            command: Command to send (up to now only "clear")
        Returns:
            Empty string
        """
        return self._putRequest(self._url("monitor","command",parameter = command), dataType = "native")

    def streamConfig(self,param = 'keys'):
        """
        Get stream configuration parameter
        Args:
            param: query the configuration parameter param, if 'keys' get all configuration parameters.
        Returns:
            If param is 'keys' return list of all parameters, else return dictionary
            that may contain the keys: value, min, max, allowed_values, unit, value_type and access_mode
        """
        return self._getRequest(self._url('stream','config',parameter = param))


    def setStreamConfig(self,param,value):
        """
        Set stream configuration parameter param.
        Args:
            param: parameter
            value: value to set
        Returns:
            List of changed parameters.
        """
        return self._putRequest(self._url('stream','config',parameter = param), dataType = 'native', data = value)

    def streamStatus(self, param):
        """Get stream status information
        Args:
            param: query the status parameter param, if 'keys' get all status parameters.
        Returns:
            Dictionary that may contain the keys: value, value_type, unit, time, state,
            critical_limits, critical_values
        """
        return self._getRequest(self._url('stream','status',parameter = param))




    #
    #
    #                Private Methods
    #
    #

    def _log(self,*args):
        if self._verbose:
            print(' '.join([ str(elem) for elem in args ]))

    def _url(self,module,task,parameter = None):
        url = f"/{self._urlPrefix}{module}/api/{self._version}/{task}/"
        if not parameter is None:
            url += f'{parameter}'
        return url

    def _getRequest(self,url,dataType = 'native', fileId = None):
        if dataType is None:
            dataType = 'native'
        if dataType == 'native':
            mimeType = 'application/json; charset=utf-8'
        elif dataType == 'tif':
            mimeType = 'application/tiff'
        elif dataType == 'hdf5':
            mimeType = 'application/hdf5'
        return self._request(url,'GET',mimeType, fileId = fileId)

    def _putRequest(self,url,dataType,data = None):
        data, mimeType = self._prepareData(data,dataType)
        return self._request(url,'PUT',mimeType, data)

    def _delRequest(self,url):
        self._request(url,'DELETE',mimeType = None)
        return None

    def _request(self, url, method, mimeType, data = None, fileId = None):
        if data is None:
            body = ''
        else:
            body = data
        headers = {}
        if method == 'GET':
            headers['Accept'] = mimeType
        elif method == 'PUT':
            headers['Content-type'] = mimeType
        if not self._user is None:
            headers["Authorization"] = f"Basic {self._user}"

        self._log(f'sending request to {url}')
        numberOfTries = 0
        response = None
        while response is None:
            try:
                self._connection.request(method,url, body = data, headers = headers)
                response = self._connection.getresponse()
            except Exception as e:
                numberOfTries += 1
                if numberOfTries == 50:
                    self._log(f"Terminate after {numberOfTries} tries\n")
                    raise e
                self._log("Failed to connect to host. Retrying\n")
                self._connection = httplibClient.HTTPConnection(self._host,self._port, timeout = self._connectionTimeout)
                continue


        status = response.status
        reason = response.reason
        if fileId is None:
            data = response.read()
        else:
            bufferSize = 8*1024
            while True:
                data = response.read(bufferSize)
                if len(data) > 0:
                    fileId.write(data)
                else:
                    break

        mimeType = response.getheader('content-type','text/plain')
        self._log('Return status: ', status, reason)
        if not response.status in range(200,300):
            raise RuntimeError((reason,data))
        if 'json' in mimeType:
            if self._serializer is None:
                return json.loads(data)
            else:
                return self._serializer.loads(data)
        else:
            return data

    def _prepareData(self,data, dataType):
        if data is None:
            return '', 'text/html'
        if dataType != 'native':
            if type(data) == 'file':
                data = data.read()
            if dataType is None:
                mimeType = self._guessMimeType(data)
                if not mimeType is None:
                    return data, mimeType
            elif dataType == 'tif':
                return data, 'application/tiff'
        mimeType = 'application/json; charset=utf-8'
        if self._serializer is None:
            return json.dumps({'value':data}), mimeType
        else:
            return self._serializer.dumps({"value":data}), mimeType

    def _guessMimeType(self,data):
        if type(data) == str:
            if data.startswith('\x49\x49\x2A\x00') or data.startswith('\x4D\x4D\x00\x2A'):
                self._log('Determined mimetype: tiff')
                return 'application/tiff'
            if data.startswith('\x89\x48\x44\x46\x0d\x0a\x1a\x0a'):
                self._log('Determined mimetype: hdf5')
                return 'application/hdf5'
        return None
