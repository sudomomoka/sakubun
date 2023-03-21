#①このプログラムは作文データを識別機に入れるために変換するプログラム

#時制 \t ['活用系'] \t ['原型'] \t 形態素解析した全文(文終わりにEOS)　\t 形態素解析、原型変換した全文 \t 何番目の単語か　
#0~5タブの形にする
#C列 A列　P列 A列　P列 C列をカウント

import pandas as pd
import openpyxl
import os


for a in range(107):#小学生女子
#for a in range(92):#小学生男子
#for a in range(100):
    #a=a+1 # 20代30代女性 小学生男女
    #a=a+101  # 40代50代女性
    #a=a+201  # 60代-80代女性
    #a=a+301  # 20代-30代男性
    #a=a+401  # 40代-50代男性
    #a=a+501  # 60代-80代男性

    #input_file_name = "../20代30代女性(ID1-100)/" + '64' + "_f.mecab.xlsx"

    #f->未来　p->過去
    #input_file_name = "../20代30代女性(ID1-100)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../40代50代女性(ID101-200)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../60-80代女性(ID201-300)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../20代30代男性(ID301-400)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../40代50代男性(ID401-500)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../60-80代男性(ID501-600)/" + str(a) + "_p.mecab.xlsx"
    #input_file_name = "../小学生女子/" + str(a) + "f_p.mecab.xlsx"
    input_file_name = "../小学生男子/" + str(a) + "m_p.mecab.xlsx"

    print("\n"+str(a)+"\n\n\n\n")

    if os.path.exists(input_file_name) == True:
        #input file name
        #input_file_name = '../20代30代女性(ID1-100)/1_f.mecab.xlsx'
        wb = openpyxl.load_workbook(input_file_name)
        ws = wb.worksheets[0]

        counter=1
        sen1 = '[CLS]'
        sen2 = '[CLS]'
        tango = []

        for row in ws.rows:#１行ずつ繰り返し
            data = [] #データ格納用リストを準備
            for cell in row:#行内のセルを繰り返し
                data.append(cell.value)#リストにセルのデータを追加
            #print(data)
            #print(data[2])
            if data[2] in ['未来','過去','現在'] :
                tango.append((data[2],data[0],data[15],counter))
            if data[0] == 'EOS':
                sen1 = sen1 + ' ' + '[SEP]' + ' ' + '[CLS]'
                sen2 = sen2 + ' ' + '[SEP]' + ' ' + '[CLS]'
                counter += 1
            else:
                sen1 = sen1 + ' ' + data[0]
                if data[15] == None:
                    sen2 = sen2 + ' ' + data[0]
                else:
                    sen2 = sen2 + ' ' + data[15]
            counter += 1
        sen1 = sen1[:-5]
        sen2 = sen2[:-5]

        #print(sen1)
        #print(sen2)
        #print(tango)

        """
        #確認
        num = tango[5]
        print(num)
        l=sen1.split()
        print(l[num[-1]])
        """

        #f->未来　p->過去
        #1-100  # 20代-30代女性
        #101-200  # 40代-50代女性
        #201-300  # 60代-80代女性
        #301-400  # 20代-30代男性
        #401-500  # 40代-50代男性
        #501-600  # 60代-80代男性

        #小学生データの改めてIDを付け直した
        #601-707 #小学生女子
        #801-900 #小学生男子

        #a=a+601  #小学生女子
        #a=a+801  #小学生男子
        save_path = "data1/" + str(a) + "_.txt"
        #save_path = "data1/" + "64" + "1_f.txt"

        with open(save_path, mode='w') as f:
            s = sen1 + '\t' + sen2
            for d in tango:
                #print(d)
                #print(type(d))
                f.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\t' + s + '\t' + str(d[3]) + '\n')


        #with open(save_path, 'a') as f:
            #for d in tango:
                #f.write(str(d) + "\n")
        f.close()
