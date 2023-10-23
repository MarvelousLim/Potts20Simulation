import re,sys
with open(sys.argv[1]) as inp:
 lines=inp.readlines()
U=[];Cull=[]
for line in lines:
 un=re.search('U_new=(.*)',line)
 if un:
  U.append(un.group(1).strip())
 cu=re.search('Culling fraction:(.*)',line)
 if cu:
  Cull.append(cu.group(1).strip())
for i in range(len(Cull)):
 print('{} {}'.format(U[i],Cull[i]))
