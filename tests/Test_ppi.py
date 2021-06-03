import winreg
import wmi
import math

PATH = "SYSTEM\\ControlSet001\\Enum\\"
m = wmi.WMI()
# 获取屏幕信息
monitors = m.Win32_DesktopMonitor()
for m in monitors:
    print(m)
    subPath = m.PNPDeviceID
    if subPath is not None:
        print(subPath)
        infoPath = PATH + subPath + "\\Device Parameters"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, infoPath)
        # 屏幕信息按照一定的规则保存（EDID）
        value = winreg.QueryValueEx(key, "EDID")[0]
        winreg.CloseKey(key)

        # 屏幕实际尺寸
        width, height = value[21], value[22]
        size = math.sqrt(pow(width, 2)+pow(height, 2)) / 2.54
        # 推荐屏幕分辨率
        widthResolution = value[56] + (value[58] >> 4) * 256
        heightResolution = value[59] + (value[61] >> 4) * 256
        sizeResolution = math.sqrt(pow(widthResolution, 2)+pow(heightResolution, 2))
        # 屏幕像素密度（Pixels Per Inch）
        widthDensity = widthResolution / (width / 2.54)
        heightDensity = heightResolution / (height / 2.54)
        sizeDensity = sizeResolution / size
        print("屏幕宽度：", width, " (厘米)")
        print("屏幕高度：", height, " (厘米)")
        print("水平分辩率: ", widthResolution, " (像素)")
        print("垂直分辩率: ", heightResolution, " (像素)")
        # 保留小数点固定位数的两种方法
        print("水平像素密度: ", round(widthDensity, 2), " (PPI)")
        print("垂直像素密度: ", "%2.f" % heightDensity, " (PPI)")
        print("PPI: ", "%2.f" % sizeDensity)
