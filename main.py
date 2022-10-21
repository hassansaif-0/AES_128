"""Input Generation Code Only Use to Populate input.pt file for testing"""
import numpy as np
import copy
import time
"""Pre Defined Constants"""
s_box = [[0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
    [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
    [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
    [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
    [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
    [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
    [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
    [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
    [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
    [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
    [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
    [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
    [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
    [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
    [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
    [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
]

inv_s_box = [
    [0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB],
    [0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB],
    [0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E],
    [0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25],
    [0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92],
    [0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84],
    [0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06],
    [0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B],
    [0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73],
    [0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E],
    [0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B],
    [0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4],
    [0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F],
    [0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF],
    [0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61],
    [0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D]
]

r_con = [
    [0x01,0x00,0x00,0x00],
    [0x02,0x00,0x00,0x00],
    [0x04,0x00,0x00,0x00],
    [0x08,0x00,0x00,0x00],
    [0x10,0x00,0x00,0x00],
    [0x20,0x00,0x00,0x00],
    [0x40,0x00,0x00,0x00],
    [0x80,0x00,0x00,0x00],
    [0x1B,0x00,0x00,0x00],
    [0x36,0x00,0x00,0x00]
    ]

"""Def"""





"""Generating The Initial Matrix from given Inputs"""
def initial_word_gen(val):
    global_word_list = []
    temlist=[]
    iter=0
    oval=0
    """Moving with step size of a word"""
    for i in range(0,len(val),8):
        temlist=[]
        copier = ""
        for k in range (0,8):
            if k %2==0 and k!=0:
                temlist.append(copier)
                copier=""
            copier=copier+val[iter]
            iter=iter+1
        temlist.append(copier)
        global_word_list.append(temlist)

    return global_word_list



"""A FUNCTION TO PERFORM 1 LEFT CIRCULAR SHIFT"""
def key_rot(inp):
    inp.append(inp.pop(0))


"""A GENERALISED SBOX SUB FUNC THAT WORK BOTH FOR SBOX AND INV SBOX"""
def sboxfun(w3,boxt):
    i=0
    j=0
    """S-BOX SUBSTITUTION"""
    for l in range(0, 4):
        """PADDING THE DATA"""
        if (len(w3[l]) < 4):
            w3[l] = w3[l][0] + w3[l][1] + '0' + w3[l][2]

        """GETTING THE APPROPRIATE ROW COLOUM FOR S BOX """
        if (w3[l][2] == '0'):
            i = 0
        elif (w3[l][2] == '1'):
            i = 1
        elif (w3[l][2] == '2'):
            i = 2
        elif (w3[l][2] == '3'):
            i = 3
        elif (w3[l][2] == '4'):
            i = 4
        elif (w3[l][2] == '5'):
            i = 5
        elif (w3[l][2] == '6'):
            i = 6
        elif (w3[l][2] == '7'):
            i = 7
        elif (w3[l][2] == '8'):
            i = 8
        elif (w3[l][2] == '9'):
            i = 9
        elif (w3[l][2] == 'a'):
            i = 10
        elif (w3[l][2] == 'b'):
            i = 11
        elif (w3[l][2] == 'c'):
            i = 12
        elif (w3[l][2] == 'd'):
            i = 13
        elif (w3[l][2] == 'e'):
            i = 14
        elif (w3[l][2] == 'f'):
            i = 15

        if (w3[l][3] == '0'):
            j = 0
        elif (w3[l][3] == '1'):
            j = 1
        elif (w3[l][3] == '2'):
            j = 2
        elif (w3[l][3] == '3'):
            j = 3
        elif (w3[l][3] == '4'):
            j = 4
        elif (w3[l][3] == '5'):
            j = 5
        elif (w3[l][3] == '6'):
            j = 6
        elif (w3[l][3] == '7'):
            j = 7
        elif (w3[l][3] == '8'):
            j = 8
        elif (w3[l][3] == '9'):
            j = 9
        elif (w3[l][3] == 'a'):
            j = 10
        elif (w3[l][3] == 'b'):
            j = 11
        elif (w3[l][3] == 'c'):
            j = 12
        elif (w3[l][3] == 'd'):
            j = 13
        elif (w3[l][3] == 'e'):
            j = 14
        elif (w3[l][3] == 'f'):
            j = 15
        w3[l] = hex(boxt[i][j])
    return i,j




"""KEY EXPANSION FUNCTION"""
def key_expansion(global_list):
    """VARIABLE FOR ROUND CONSTANT FETCHING"""
    iter=0
    for u in range(0,len(global_list)):
        for k in range(0,4):
            global_list[u][k]=hex(int(global_list[u][k],16))

    """ GENERATING 40 WORDS OR 10 KEYS"""
    for g in range(0,40,4):
        temlist=[]
        """INITIAL WORDS TO BEGIN WITH IN EACH ROUND"""
        w0=global_list[g]
        w1=global_list[g+1]
        w2=global_list[g+2]
        w3=global_list[g+3]

        """COPYING TO MAKE THE DATA SAFE"""
        w3_init=copy.deepcopy(global_list[g+3])


        """Rotation Function"""
        key_rot(w3)

        i,j=sboxfun(w3,s_box)


        """Result Xoring for yielding g func"""
        gfun = []
        """CALCULATING G FUNCTION"""
        for m in range(0,4):
            gfun.append(hex(int(w3[m],16)^int(hex(r_con[iter][m]),16)))

        """CALCULATION OF REMAINING WORDS"""
        for n in range(0,4):
            temlist.append(hex(int(w0[n],16)^int(gfun[n],16)))
        w4=temlist

        temlist=[]
        for n in range(0,4):
            val1=int(w1[n],16)
            val2=int(w4[n],16)
            temlist.append(hex(val1^val2))
        w5 = temlist

        temlist = []
        for n in range(0,4):
            temlist.append(hex(int(w2[n],16)^int(w5[n],16)))
        w6 = temlist

        temlist = []

        for n in range(0,4):
            temlist.append(hex(int(w3_init[n],16)^int(w6[n],16)))
        w7 = temlist


        for con in range(0,4):
            w4[con]=str(w4[con])
            w5[con]=str(w5[con])
            w6[con]=str(w6[con])
            w7[con]=str(w7[con])

        """APPENDING THE RESULTS FOR FURTHER KEY EXPANSION"""
        global_list.append(w4)
        global_list.append(w5)
        global_list.append(w6)
        global_list.append(w7)
        global_list[g+3]=w3_init

        iter+=1

    return global_list


"""Initial Round Key Addup"""
def init_set(text,global_list):
    for k in range(0,4):
        for r in range(0,4):
            text[k][r]=hex(int(text[k][r],16)^int(global_list[k][r],16))
    return text


"""Initial Round Key Addup for Decryption"""
def inv_init_set(ciphtext,keylist):
    start=len(keylist)-4

    for k in range(0,4):
        for r in range(0,4):
            ciphtext[k][r]=hex(int(ciphtext[k][r],16)^int(keylist[start][r],16))
        start=start+1
    return ciphtext

"""A MULTIPLICATION FUNCTION ACCORDING TO GALLOIS FIELD RULES"""
def mul(first, second):
    a = first
    b = second
    product = 0
    for i in range(8):
        if (b & 1) == 1:
            product ^= a
        hi_bit_set = a & 0x80
        a = (a << 1) & 0xFF
        if hi_bit_set == 0x80:
            a ^= 0x1B
        b >>= 1
    return product

"""AES ENCRYPTION METHOD"""
def AES_ENC(text_list,keylist):
    for i in range (0,9):
        print("\n\n=============================Round==",i+1,"==Going==================================")
        """S BOX SUB"""
        for j in range(0,4):
            sboxfun(text_list[j],s_box)
        print("================================AFTER S BOX=============================================\n",text_list)

        """SHIFT ROW"""
        check=np.array(text_list)
        check=check.transpose()

        for k in range(1,4):
            check[k]=np.roll(check[k],-k)

        print("================================AFTER ROW SHIFT=========================================\n",check)

        original=copy.deepcopy(check)

        """Mixing Coloumn"""
        for n in range(0,4):
            check[0][n]=hex((mul(2,int(original[0][n],16)) )^(mul(3,int(original[1][n],16)))^int(original[2][n],16)^int(original[3][n],16))
            check[1][n]=hex(int(original[0][n],16) ^ (mul(2,int(original[1][n], 16))) ^ (mul(3,int(original[2][n],16))) ^ int(original[3][n], 16))
            check[2][n]=hex(int(original[0][n],16)^(int(original[1][n],16)) ^ (mul(2,int(original[2][n], 16))) ^(mul(3,int(original[3][n],16))))
            check[3][n]=hex(mul(3,int(original[0][n],16))^(int(original[1][n], 16)) ^ (int(original[2][n],16)) ^ (mul(2,int(original[3][n], 16))))

        check=check.transpose()
        check=check.tolist()

        print("================================after col mix===========================================\n",check)

        """AFTER ROUND KEY"""
        init_set(check,wlist)
        print("================================After the round key addup===============================\n",check)
        for z in range(0,4):
            keylist.pop(0)
        text_list=check
        print("\n\n")

    print("====================================================Round==",10, "==Going==========================")
    """S BOX SUB"""
    for j in range(0, 4):
        sboxfun(text_list[j],s_box)
    print("===================================================AFTER S BOX====================================\n", text_list)

    """SHIFT ROW"""
    check = np.array(text_list)
    check = check.transpose()
    for k in range(1, 4):
        check[k] = np.roll(check[k], -k)
    print("==================================================AFTER ROW SHIFT=================================\n", check)
    check = check.transpose()
    check = check.tolist()
    """AFTER ROUND KEY"""
    init_set(check, wlist)
    print("=================================================After the round key addup========================\n", check)

    print("\n\n\n")
    return check



"""AES DECRYPTION METHOD"""
def AES_DEC(text_list,keylist):
    for i in range (0,9):

        """SHIFT ROW"""
        check=np.array(text_list)
        check=check.transpose()
        for k in range(1,4):
            check[k]=np.roll(check[k],k)

        print("=====================================AFTER ROW SHIFT===========================================\n",check)

        """Inverse S Box"""
        print("\n==================================Round=",i+1,"=Going===============================================")
        check=check.transpose()
        check=check.tolist()
        """S BOX SUB"""
        for j in range(0,4):
            sboxfun(check[j],inv_s_box)
        print("====================================AFTER INVERSE S BOX=======================================\n",check)

        """ROUNDKEY ADDUP"""
        inv_init_set(check,keylist)
        print("===================================After the round key========================================\n", check)


        """INVERSEMIXCOL"""
        check = np.array(check)
        check=check.transpose()
        original=copy.deepcopy(check)

        for n in range(0,4):

            check[0][n]=hex((mul(0xE,int(original[0][n],16)) )^(mul(0xB,int(original[1][n],16)))^mul(0xD,int(original[2][n],16))^mul(0x9,int(original[3][n],16)))
            check[1][n]=hex(mul(0x9,int(original[0][n],16)) ^ (mul(0xE,int(original[1][n], 16))) ^ (mul(0xB,int(original[2][n],16))) ^ mul(0xD,int(original[3][n], 16)))
            check[2][n]=hex(mul(0XD,int(original[0][n],16))^mul(0x9,(int(original[1][n],16))) ^ (mul(0xE,int(original[2][n], 16))) ^(mul(0xB,int(original[3][n],16))))
            check[3][n]=hex(mul(0xB,int(original[0][n],16))^mul(0xD,(int(original[1][n], 16))) ^ mul(0x9,(int(original[2][n],16))) ^ (mul(0xE,int(original[3][n], 16))))
        print("=================================================After MixCol================================\n",check)




        check=check.transpose()
        check=check.tolist()
        start = len(keylist) - 4
        for i in range(0, 4):
            keylist.pop(start)
        text_list=check

    check = np.array(text_list)
    check = check.transpose()
    for k in range(1, 4):
        check[k] = np.roll(check[k], k)

    print("=================================================AFTER ROW SHIFT==================================\n", check)

    """Inverse S Box"""
    print("================================================Round=",10, "=Going=======================================")
    check = check.transpose()
    check = check.tolist()
    """S BOX SUB"""
    for j in range(0, 4):
        sboxfun(check[j], inv_s_box)
    print("============================================AFTER INVERSE S BOX===================================\n", check)

    """ROUNDKEY ADDUP"""
    inv_init_set(check, keylist)
    print("============================================After the round key==================================\n", check)
    return check


"""FUNCTIONCALLINGS STARTS FROM HERE"""

# Reading the initial files for initializations
key_0 = open("inpkey.key", "r")
textval=""
text=open("input.pt","r")

val=""
val=key_0.read()
textval=text.read()

textval=textval.replace("\n","")
if(len(textval)<32):
    for i in range(len(textval),32):
        textval=textval+"0"

val=val.replace("\n","")
if(len(val)<32):
    for i in range(len(val),32):
        val=val+"0"


print("Input Text is:",textval)
print("Initial key is:",val)

wlist=initial_word_gen(val)
textlist=initial_word_gen(textval)



"""KEY EXPANSIONS"""
start = time.time()
key_expansion(wlist)

print("Expanded keys words list are::",wlist)

"""DEEP COPY FOR SECURING DATA"""
original_keys=copy.deepcopy(wlist)
deckeys=copy.deepcopy(wlist)


"""PRE ROUND KEY ADDUP"""
textlist=init_set(textlist,wlist)
"""creating a copy"""
temtextlist=copy.deepcopy(textlist)


"""Removing the pre round keys"""
for i in range (0,4):
    wlist.pop(0)


ciphertext=AES_ENC(textlist,wlist)

myfile=open("encryption.enc","w")
for i in range(0,4):
    for j in range(0,4):
        val=ciphertext[i][j]
        if(len(val)<4):
            val="0"+val[2]
        else:
            val=val[2]+val[3]
        myfile.write(val)
myfile.close()


temcipher=copy.deepcopy(ciphertext)





print("Our Cipher is::",ciphertext)


"""Decryption Part"""
ciphlist=inv_init_set(temcipher,deckeys)
print("After add Round Key10\n",ciphlist)

"""pre round key removal"""
start=len(deckeys)-4

for i in range (0,4):
    deckeys.pop(start)

plaintext=AES_DEC(ciphlist,deckeys)

myfile=open("decryption.dec","w")
for i in range(0,4):
    for j in range(0,4):
        val=plaintext[i][j]
        if(len(val)<4):
            val="0"+val[2]
        else:
            val=val[2]+val[3]
        myfile.write(val)
myfile.close()


print("After performing decryption we got ::",plaintext)






