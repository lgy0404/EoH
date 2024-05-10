#-----------------deepseek-----------------##
from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "tsp_construct", #['tsp_construct','bp_online']
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = "sk-33a981766e724cf4880e0fe3729534b5",   # set your key
                llm_model = "deepseek-chat",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()


##-----------------OPRNAI-----------------##
# from eoh import eoh
# from eoh.utils.getParas import Paras

# # Parameter initilization #
# paras = Paras() 

# # Set parameters #
# paras.set_paras(method = "eoh",    # ['ael','eoh']
#                 problem = "tsp_construct", #['tsp_construct','bp_online']
#                 llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
#                 llm_api_key = "sk-zKs3eVUnmpT281l1bs6CdFPUaabv0ocMSonmH9FekGftF5hZ",   # set your key
#                 llm_model = "gpt-3.5-turbo",
#                 ec_pop_size = 4, # number of samples in each population
#                 ec_n_pop = 4,  # number of populations
#                 exp_n_proc = 4,  # multi-core parallel
#                 exp_debug_mode = False)

# # initilization
# evolution = eoh.EVOL(paras)

# # run 
# evolution.run()

##-----------------ORIGIN-----------------##
# from eoh import eoh
# from eoh.utils.getParas import Paras

# # Parameter initilization #
# paras = Paras() 

# # Set parameters #
# paras.set_paras(method = "eoh",    # ['ael','eoh']
#                 problem = "tsp_construct", #['tsp_construct','bp_online']
#                 llm_api_endpoint = "XXX", # set your LLM endpoint
#                 llm_api_key = "XXX",   # set your key
#                 llm_model = "gpt-3.5-turbo",
#                 ec_pop_size = 4, # number of samples in each population
#                 ec_n_pop = 4,  # number of populations
#                 exp_n_proc = 4,  # multi-core parallel
#                 exp_debug_mode = False)

# # initilization
# evolution = eoh.EVOL(paras)

# # run 
# evolution.run()