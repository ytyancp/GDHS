# Always consider square confusion matrices as follows:
#             preds
#           c1 c2 c3
#       c1  5   2   0
# trues c2 10  200  1
#       c3  3   5  300
#

#
# mat <- matrix(c(5,2,0,10,200,1,3,5, 300), nrow=3, ncol=3, byrow=TRUE)
# colnames(mat) <- c("c1", "c2", "c3")
# rownames(mat) <- c("c1", "c2", "c3")

#R语言转python
#矩阵z
import numpy as np
import math
## Average Accuracy

def search_index (l, a):
#找到a在l中的所有位置返回列表
    list = []
    for i in range(len(l)):
        if a == l[i]:
            list.append(i)
    return list

def rowsums(z):#求矩阵行和 返回列表 长度与行数相等
    list = []
    row_num = z.shape[0]
    col_num = z.shape[1]
    for i in range(row_num):
        sum = 0
        for j in range(col_num):
            sum = sum +z[i][j]
        list.append(sum)
    return list

def colsums(z):#求矩阵列和 返回列表 长度与列数相等
    list = []
    row_num = z.shape[0]
    col_num = z.shape[1]
    for i in range(col_num):
        sum = 0
        for j in range(row_num):
            sum = sum +z[j][i]
        list.append(sum)
    return list

def MAUC(z):
    res = 0
    corr = np.diagonal(z)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i < j:
                if z[i][i] + z[i][j] ==0 or z[j][i] + z[j][j]==0:
                    auc1 = 0
                else:
                    tprate =  z[i][i] / (z[i][i] + z[i][j])
                    fprate =  z[j][i] / (z[j][i] + z[j][j] )
                    auc1 = (1 + tprate - fprate) / 2
                if z[j][j] + z[j][i] ==0 or z[i][j] + z[i][i] ==0:
                    auc2= 0
                else:
                    tprate = z[j][j] / (z[j][j] + z[j][i])
                    fprate = z[i][j] / (z[i][j] + z[i][i])
                    auc2 = (1 + tprate - fprate) / 2
                res = res + (auc1 + auc2)/2
    return float(2*res/(len(corr)*(len(corr)-1)))

def AvACC(z):
    corr = np.diagonal(z)
    t = rowsums(z)
    p = colsums(z)
    res = 0
    for i in range(len(corr)):
        if sum(corr)+t[i]+p[i]-2 * corr[i] ==0:
            return None
        else:
            res = res + (sum(corr) / (sum(corr)+t[i]+p[i]-2 * corr[i]))
    res = float(res/len(corr))
    return res

## Macro Average Geometric (geometric average of each class recall)
def MAvG(z):
    corr = np.diagonal(z)
    t= rowsums(z)
# corr: correclty classified
# t: total trues of each class
    res = 1
    for i in range(len(corr)):
        if t[i]==0:
            return None
        else:
            res = res * corr[i]/t[i]
    res = float ( res**(1/len(corr)))
    return res

# Macro Average Arithmetic / Recall macro
def MAvA(z):
    corr = np.diagonal(z)
    t = rowsums(z)
    res = 0
    for i in range(len(corr)):
        res = res + corr[i]/t[i]
    res = float (res/len(corr))
    return res

## Precision macro
def precM(z):
    corr = np.diagonal(z)
    p = colsums(z)
    res = 0
    index = search_index (p, 0)
    if len(index) == len(p):
        res = None
    else:
        for i in range(len(p)):
            if i not in index:
                res = res +corr[i]/p[i]
    res = float(res/len(corr))
    return res

## Recall micro
def recMiu(z):
    corr = np.diagonal(z)
    t =rowsums(z)
    res = sum(corr)/sum(t)
    return res

## Precision micro
def precMiu(z):
    corr = np.diagonal(z)
    p = colsums(z)
    res = sum(corr)/sum(p)
    return res

## MFb (Alejo and Ferri 2009 extended for any beta)
def MFb(beta ,z):
    corr = np.diagonal(z)
    t = rowsums(z)
    p = colsums(z)
    res = 0
    for i in range(len(corr)):
        res = res+(((1+beta ** 2) * corr[i]) / ((beta **2 )* t[i]+p[i]))# robusteness for indefined precision or recall in one class
    res = float(res/len(corr))
    return res


## MFbM (sokolova 2009)
def Fbm(beta,z):
    rec = MAvA(z)
    prec = precM(z)
    res = float( ((1 + beta ** 2) * prec * rec) /( (beta ** 2) * prec + rec))
    return res

## MFbMiu (sokolova 2009)
def FbMiu(beta,z):
    rec = recMiu(z)
    prec = precMiu(z)
    res = ((1 + beta ** 2) * prec * rec) / ((beta ** 2 )* prec + rec)
    return res

### CBA
def across(u,v, t):
    if sum(u) == 0 and sum(v) == 0:
        return 0
    else:
        return t / max(sum(u), sum(v))


def cba(z):
    n = z.shape[0]
    xyacross =  np.full((n,n),0.0)
    for i in range(n):
        for j in range(n):
            xyacross[i][j] = across(z[i], z[:,j],z[i][j])
    return np.mean(np.diagonal(xyacross))

def MCC(z):
    n = z.shape[0]
    nom = 0
    for k in range(n):
        for l in range(n):
            for m in range(n):
                nom = nom + (z[k][k] * z[m][l] - z[l][k] * z[k][m])
    den1 = 0
    den2 = 0
    for k in range(n):
        term1 = 0
        term2 = 0
        term3 = 0
        term4 = 0
        for l in range(n):
            term1 = term1 + z[l][k]
            term3 = term3 + z[k][l]
        for f in range(n):
            if f != k:
                for g in range(n):
                    term2  = term2 + z[g][f]
                    term4 = term4 + z[f][g]
        den1 = den1 + term1*term2
        den2 = den2 + term3*term4
    if den1 == 0 or den2==0:
        return None
    else:
        res = nom / ((den1**0.5 )* (den2**0.5))
    return res

def CENClass(z,j):
    n = z.shape[0]
    rs = rowsums(z)
    cs = colsums(z)
    probs = np.full((n,n),0.0)# probs(i,k) = prob of classifying class i to class k subject to class j.
    for k in range(n):
        if k != j:
            sum1 = rs[j]
            sum2 = cs[j]
            # p_jk
            probs[j][k] = float(z[j][k] / (sum1+ sum2))
            # p_kj
            probs[k][j] = float(z[k][j] / (sum1 + sum2))
        else:
            # p_jj
            probs[j][j] = 0
    if (sum1 + sum2) == 0:
        res = 0
        return res
    else:
        res = 0
        for k in range(n):
            if k!=j:
                l1=0
                l2=0
                if probs[j][k]!=0:
                    l1 = probs[j][k] * math.log(probs[j][k], 2 * (n-1))
                if probs[k][j]!=0:
                    l2 = probs[k][j] * math.log(probs[k][j], 2 * (n - 1))
                res = res+l1+l2
    return -res
def sumres(z):
    sumrevalue = 0
    for i in range(len(z[0])):
        for j in range(len(z[0])):
            sumrevalue = sumrevalue + z[i][j]
    return sumrevalue
def CEN(z):
    n =z.shape[0]
    rs = rowsums(z)
    cs = colsums(z)
    res = 0
    for j in range(n):
        P = (rs[j] + cs[j]) / (2 * sumres(z))
        res = res + P * CENClass(z, j)
    res = float(res)
    return res

###############################################################################
## Weighted measures: the relevance is AUTOMATICALLY determined
## using the classes frequency
## strategy labeled as PRE in the paper; here appended with W
###############################################################################

def wEvt(t):
 #   res1 = 1/t
    res1 = []
    for i in t:
        res1.append(1 / i)
        # res1.append(1/float(i))
    res = []
    for i in res1:
        res.append(i/sum(res1))
 #   res = res1 /sum(res1)
    return res
# wEvt < - function(t)
# {
# input: t: vector with original frequencies (trues) of each class
# output: vector with normalized weights for each class
# res1 < - 1 / t
# res < - res1 / sum(res1)
# res
# }

def wEvp(p):
    res1 = []
    for i in p:
        res1.append(1/i)
   # res1 = 1/p
    res = []
    for i in res1:
        res.append(i/sum(res1))
#    res = res1/(sum(res1))
    return res
# wEvp < - function(p)
# {
# # input: p: vector with predicted frequencies (preds) of each class
# # output: vector with normalized weights for each class
# res1 < - 1 / p
# res < - res1 / sum(res1)
# res
# }

def WMRec(z):
    corr = np.diagonal(z)
    t = rowsums(z)
    wi = wEvt(t)
    res = 0
    for i in range(len(t)):
        res =res + wi[i] *corr[i]/t[i]
    res = float(res)
    return res


def WMPrec(z):
    corr=np.diagonal(z)
    t = rowsums(z)
    p = colsums(z)
    wi = wEvt(t)
    res = 0
    index = search_index(p,0)
    if len(index)== len(p):
        return None
    else:
        k = 0
        for i in range(len(p)):
            if i not in index:
                k = k + wi[i]
        wi = wi/k
        for i in range(len(p)):
            if i not in index:
                res = res + wi[i] * corr[i] / p[i]
    res = float(res)
    return res

# weighted Fmeasure with previous calculated weighted rec and prec
def WFM(beta,z):
    prec = WMPrec(z)
    rec = WMRec(z)
    res = float (((1 + beta ** 2) * prec * rec) / ((beta ** 2) * prec + rec))
    return res

# weighted averaged FM
def WAvFM(beta,z):
    corr = np.diagonal(z)
    t =rowsums(z)
    p = colsums(z)
    res = 0
    wit = wEvt(t)
    for i in range(corr):
        res = res + (wit[i] * (1+beta **2) * corr[i]) / ((beta ** 2 )* t[i]+p[i])  # robust solution for undefined recall or
    res = float(res)
    return res


# cba with phi weighting automatic
def across2(u,v,t):
    if sum(u)==0 and sum(v)==0:
        return 0
    else:
        tvalue = float(t/max(sum(u),sum(v)))
    return tvalue

def Wcba(z):
    tr = rowsums(z)
    wi =wEvt(tr)
    n = z.shape[0]
    xyacross =  np.full((n,n),0.0)
    for i in range(n):
        for j in range(n):
            xyacross[i][j] = across2(z[i], z[:, j], z[i, j])
    return sum(wi*np.diagonal(xyacross))


