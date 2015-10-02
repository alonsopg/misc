# -*- coding: utf-8 -*-

import os,sys
folder = '/Users/user/Desktop/OpinionsTag'
for filename in os.listdir(folder):
       infilename = os.path.join(folder,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('.tag', '.txt')
       output = os.rename(infilename, newname)