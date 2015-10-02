# -- coding: utf-8 --
import re


str = "pero pero CC " \
      "tan tan RG " \
      "antigua antiguo AQ0FS0" \
      "antigua antiguo AQ1FS0"\
      "que que CS " \
      "según según SPS00 " \
      "mi mi DP1CSS " \
      "madre madre NCFS000"



tupla1 = re.findall(r'(\w+)\s\w+\s(AQ)', str)


tupla2 = re.findall(r'(\w+)\s\w+\s(NCFS00)',str)
print (tupla1 + tupla2)