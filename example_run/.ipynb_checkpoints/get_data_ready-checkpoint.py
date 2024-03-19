import os, csv, sys, glob, shutil


folder = '.'

if os.path.isdir('%s/formatted_data'%folder):
    shutil.rmtree('%s/formatted_data'%folder)
os.mkdir('%s/formatted_data'%folder)



peps = []
tcrs = []
with open('%s/data/train.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))
with open('%s/data/val.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))
with open('%s/data/test.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))

peps = list(set(peps))
tcrs = list(set(tcrs))
pep_seq_to_id = {}
tcr_seq_to_id = {}

with open('%s/formatted_data/ids_pep.csv'%folder, 'w') as fw:
    for en, pep in enumerate(peps):
        fw.write('pep%d,%s\n'%(en,pep))
        pep_seq_to_id[pep] = 'pep%d'%en
        
with open('%s/formatted_data/ids_tcr.csv'%folder, 'w') as fw:
    for en, tcr in enumerate(tcrs):
        fw.write('tcr%d,%s\n'%(en,tcr))
        tcr_seq_to_id[tcr] = 'tcr%d'%en
    
    
    
        

peps, tcrs, labels, split = [],[],[],[]
    
with open('%s/data/train.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('train')

with open('%s/data/val.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('val')

with open('%s/data/test.csv'%folder, 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('test')
    
with open('%s/formatted_data/data.csv'%folder, 'w') as fw:
    fw.write('pep_id,tcr_id,label,split\n')
    for aa,bb,cc,dd in zip(peps, tcrs, labels, split):
        fw.write('%s,%s,%d,%s\n'%(aa,bb,cc,dd))



        