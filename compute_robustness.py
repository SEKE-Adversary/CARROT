
def parse_log(log_file_path):
	with open(log_file_path, "r") as f:
		lines = f.readlines()

	# NEW VERSION FILTER (More Accurate)
	lines_ = []
	for line in lines:
		if line.lstrip().startswith("acc") or line.lstrip().startswith("rej") or line.lstrip().startswith("skip unk") or \
		   line.lstrip().startswith("FAIL!") or line.lstrip().startswith("SUCC!") or line.lstrip().startswith("FATAL ERROR!") or \
		   line.lstrip().startswith("Curr succ rate") or ("ID = " in line and "Y = " in line):
		   lines_.append(line)
	lines = lines_
	
	# Get partial log for each test sample
	parts = []
	tmp_part = []
	for line in lines:
		tmp_part.append(line)
		if line.lstrip().startswith("Curr succ rate"):
			parts.append(tmp_part)
			tmp_part = []

	# Collect ID2Res dict, value options: ["FAIL", "SUCC_ATK", "SUCC_ORI"]
	ID2Res = {}
	for part in parts:
		info_line = part[0]
		i = 1
		res_line = part[i]
		while res_line.startswith("rej") or res_line.startswith("acc") or res_line.startswith("skip unk"):
			i += 1
			res_line = part[i]
		Id = info_line.strip().split("\t")[1]
		if res_line.startswith("FAIL!") or res_line.startswith("FATAL ERROR!"):
			res = "FAIL"
		elif res_line.startswith("SUCC!\t"):
			res = "SUCC_ATK"
		elif res_line.startswith("SUCC! Original mistake."):
			res = "SUCC_ORI"
		else:
			assert False, res_line
		ID2Res[Id] = res
		'''	# for debug
		print("---------------------")
		for line in part:
			print(line, end="")
		print("[%s => %s]" % (Id, res))
		'''
	return ID2Res

def get_dual_fail_IDs(stmt_log_filepath, uid_log_filepath):
	ID2Res_stmt = parse_log(stmt_log_filepath)
	ID2Res_uid = parse_log(uid_log_filepath)
	
	fail_IDs = []
	IDs = list(set(ID2Res_stmt.keys()) & set(ID2Res_uid.keys()))
	for Id in IDs:
		if ID2Res_stmt[Id]=="FAIL" and ID2Res_uid[Id]=="FAIL":
			fail_IDs.append(Id)
	return fail_IDs, list(ID2Res_stmt.keys())

def get_single_fail_IDs(log_filepath):
	ID2Res = parse_log(log_filepath)

	fail_IDs = []
	for Id in ID2Res.keys():
		if ID2Res[Id]=="FAIL":
			fail_IDs.append(Id)
	return fail_IDs, list(ID2Res.keys())

if __name__=="__main__":

    # Compute robustness with the log files
    # This is an example
    # Suppose the attack log of I-CARROT is ``id.log''
    #   and the attack log of S-CARROT is ``stmt.log''
    # Then run the following commands

	parser = argparse.ArgumentParser()
    parser.add_argument('-I', type=str, required=True)
    parser.add_argument('-S', type=str, required=True)
    parser.add_argument('-model', type=str, default="MODEL")
    
    opt = parser.parse_args()

    victim_model = opt.model
    I_log = opt.I
    S_log = opt.S
    print(victim_model, end=" ")
    fail_IDs, IDs = get_dual_fail_IDs(S_log, I_log)
    n_fail, n_id = len(fail_IDs), len(IDs)
    rate = n_fail/float(n_id)
    print(str(n_fail)+"/"+str(n_id), "=> %.4f" % rate)
