import os,sys,signal,ctypes
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'prctl PR_SET_PDEATHSIG')
if os.getppid()==1:raise SystemExit('controller exited before child attached')
os.execv(sys.executable,[sys.executable,'-u',sys.argv[1]])
