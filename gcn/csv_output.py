import csv

def csv_output(path, table):
    f = open(path,'w',encoding='utf-8', newline='' "")

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 4. 写入csv文件内容
    for i in table:
        csv_writer.writerow(i)

    # 5. 关闭文件
    f.close()