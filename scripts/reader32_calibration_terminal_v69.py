"""Fixed terminal fields only; no raw exception/error text."""
import sys,time
from pathlib import Path
from reader32_calibration_contract_v69 import write_metadata

def main():
    spool=Path(sys.argv[1]);status=int(sys.argv[2])
    write_metadata(spool/'calibration_terminal.json','terminal',{'kind':3,'exit_code':status,'at':time.time(),'calibration_only':True,'qualification_established':False,'model_calls':0,'GPU_allocations':0})
if __name__=='__main__':main()
