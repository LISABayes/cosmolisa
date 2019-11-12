import os
from optparse import OptionParser

MBH_Models = ['heavy_Q3',
              'heavy_no_delays',
              'popIII']

#EMRI_Models = ['EMRI_M1_GAUSS']

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option('-p','--program',default=None,type='string',help='Executable')
    parser.add_option('--path',default=None,type='string',help='Executable path')
    parser.add_option('-t','--threads',default=4,type='int',metavar='threads',help='Number of cpu cores to request (default = all of them)')
    parser.add_option('-d','--data',default=None,type='string',metavar='data',help='catalogs location (folder with EVENT_ subfolders)')
    parser.add_option('-e','--event',default=None,type='int',metavar='event',help='event number')
    parser.add_option('-w','--work',default=None,type='string',metavar='work',help='base folder containing the catalogs, the jobs will be executed there')
    parser.add_option('-c','--source-class', default=None, type='string', metavar='source_class', help='source class to analyse')
    parser.add_option('-m','--model', default='LambdaCDM', type='string', metavar='model', help='cosmological model to assume (default: LambdaCDM). Supports LambdaCDM and LambdaCDMDE')
    parser.add_option('-j','--joint', default=0, type='int',metavar='joint',help='run a joint analysis for N events, randomly selected. (EMRI only)')
    parser.add_option('--zhorizon', default=1.0, type='float',metavar='zhorizon',help='Horizon redshift corresponding to the SNR threshold')
    parser.add_option('--snr_threshold', default=20, type='int',metavar='snr_threshold',help='SNR selection threshold')
    parser.add_option('--gw_selection', default=0, type='int',metavar='gw_selection',help='use GW selection function')
    parser.add_option('--em_selection', default=1, type='int',metavar='em_selection',help='use EM selection function')
    parser.add_option('--submit', default=0, type='int',metavar='submit',help='automatically submit the job (default: False)')
    
    (opts,args)=parser.parse_args()
    
    program = opts.program
    submit  = opts.submit
    executable = os.path.join(opts.path,program)
    if opts.source_class =='MBH':
        for M in MBH_Models:
            work_folder = os.path.join(opts.work,"%s"%M)
            all_files = os.listdir(work_folder)
            catalogs = [f for f in all_files if 'cat' in f]
            for C in catalogs:
                catalog_folder = os.path.join(work_folder,"%s"%C)
                os.system("mkdir -p %s"%os.path.join(catalog_folder,"log"))
           
                submit_file = open(os.path.join(catalog_folder,"submit_file.sub"),"w")
                submit_file.write("Universe = vanilla\n")
                submit_file.write("Executable = %s\n"%executable)
                submit_file.write("output = %s/log/%s_out.$(Process)\n"%(catalog_folder,program))
                submit_file.write("error = %s/log/%s_err.$(Process)\n"%(catalog_folder,program))
                submit_file.write("log = %s/log/%s_log.$(Process)\n"%(catalog_folder,program))
                submit_file.write("notification = Never\n")
                submit_file.write("getEnv = True\n")
                submit_file.write("request_cpus = %d\n"%opts.threads)
                submit_file.write("\n")
                submit_file.write("arguments = -d %s -o %s -m %s -c %s\n"%(catalog_folder,catalog_folder,opts.model,opts.source_class))
                submit_file.write("queue 1\n")
                submit_file.write("\n")
                submit_file.close()
                if submit: os.system('condor_submit %s'%os.path.join(catalog_folder,"submit_file.sub"))
    elif opts.source_class == 'EMRI':
#        for M in EMRI_Models:
        work_folder = os.path.join(opts.work) #,"%s"%M)
        if opts.joint == 0:
            all_files = os.listdir(work_folder)
            events = [e for e in all_files if 'EVENT' in e]
            for E in events:
                event_folder = os.path.join(work_folder,"%s"%E)
                os.system("mkdir -p %s"%os.path.join(event_folder,"log"))
                submit_file = open(os.path.join(event_folder,"submit_file.sub"),"w")
                submit_file.write("Universe = vanilla\n")
                submit_file.write("Executable = %s\n"%executable)
                submit_file.write("output = %s/log/%s_out.$(Process)\n"%(event_folder,program))
                submit_file.write("error = %s/log/%s_err.$(Process)\n"%(event_folder,program))
                submit_file.write("log = %s/log/%s_log.$(Process)\n"%(event_folder,program))
                submit_file.write("notification = Never\n")
                submit_file.write("getEnv = True\n")
                submit_file.write("request_cpus = %d\n"%opts.threads)
                submit_file.write("\n")
                J = int(E.split('_')[-1][1:])-1
                submit_file.write("arguments = -d %s -o %s -e %d -m %s -c %s --zhorizon %s --gw_selection %d --em_selection %d --snr_threshold %d\n"%(work_folder,event_folder,J,opts.model,opts.source_class,opts.zhorizon,opts.gw_selection,opts.em_selection,opts.snr_threshold))
                submit_file.write("queue 1\n")
                submit_file.write("\n")
                submit_file.close()
                if submit: os.system('condor_submit %s'%os.path.join(event_folder,"submit_file.sub"))
        else:
            catalog_folder = os.path.join(work_folder,"catalog_%d"%opts.joint)
            os.system("mkdir -p %s"%os.path.join(catalog_folder,"log"))
            submit_file = open(os.path.join(catalog_folder,"submit_file.sub"),"w")
            submit_file.write("Universe = vanilla\n")
            submit_file.write("Executable = %s\n"%executable)
            submit_file.write("output = %s/log/%s_out.$(Process)\n"%(catalog_folder,program))
            submit_file.write("error = %s/log/%s_err.$(Process)\n"%(catalog_folder,program))
            submit_file.write("log = %s/log/%s_log.$(Process)\n"%(catalog_folder,program))
            submit_file.write("notification = Never\n")
            submit_file.write("getEnv = True\n")
            submit_file.write("request_cpus = %d\n"%opts.threads)
            submit_file.write("\n")
            submit_file.write("arguments = -d %s -o %s -m %s -c %s -j %d\n"%(work_folder,catalog_folder,opts.model,opts.source_class,opts.joint))
            submit_file.write("queue 1\n")
            submit_file.write("\n")
            submit_file.close()
            if submit: os.system('condor_submit %s'%os.path.join(catalog_folder,"submit_file.sub"))
