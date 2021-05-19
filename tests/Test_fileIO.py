fo = open("../res/foo.txt", "w")
dict = {(1,3):(3,4),(1,):(10,11)}
for d in dict.keys():
    fo.write('%s:%s\n' % (d,dict[d]))


# 关闭打开的文件
fo.close()