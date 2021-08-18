import winreg
import wmi
import math


class ScreenHelper:
    """获取屏幕相关的信息

    """
    def __init__(self):
        PATH = "SYSTEM\\ControlSet001\\Enum\\"
        m = wmi.WMI()
        # 获取屏幕信息
        monitors = m.Win32_DesktopMonitor()
        for m in monitors:
            subPath = m.PNPDeviceID
            if subPath is not None:
                infoPath = PATH + subPath + "\\Device Parameters"
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, infoPath)
                # 屏幕信息按照一定的规则保存（EDID）
                value = winreg.QueryValueEx(key, "EDID")[0]
                winreg.CloseKey(key)
                # 屏幕实际尺寸
                width, height = value[21], value[22]
                size = math.sqrt(pow(width, 2) + pow(height, 2)) / 2.54
                self.size = size  #屏幕实际尺寸
                # 推荐屏幕分辨率
                widthResolution = value[56] + (value[58] >> 4) * 256
                heightResolution = value[59] + (value[61] >> 4) * 256
                self.widthResolution = widthResolution
                self.heightResolution = heightResolution
                sizeResolution = math.sqrt(pow(widthResolution, 2) + pow(heightResolution, 2))  #对角线的分辨率
                # 屏幕像素密度（Pixels Per Inch）
                if size != 0:
                    sizeDensity = sizeResolution / size
                self.PPI = sizeDensity

    def getHResolution(self):
        return self.heightResolution

    def getWResolution(self):
        return self.widthResolution

    def getPPI(self):
        return self.PPI

    def getSize(self):
        return self.size