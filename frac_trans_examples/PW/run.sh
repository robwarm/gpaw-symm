#!/bin/bash
#gpaw-python quartz_lft.py > aq_current.out
#gpaw-python quartz.py > aq_ref_current.out
#gpaw-python quartz_nosym.py > aq_refnosym_current.out

gpaw-python stishovite_lft.py > s_current.out
gpaw-python stishovite.py > s_ref_current.out
gpaw-python stishovite_nosym.py > s_refnosym_current.out

