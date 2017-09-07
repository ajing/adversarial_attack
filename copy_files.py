import os
import shutil
import subprocess

DEST="/home/ubuntu/cleverhans/examples/nips17_adversarial_competition"
META_DIR = "/home/ubuntu/adversarial_attack/metafiles"
CONFIG_DIR = "config.csv"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# organize the files based on config.csv
all_content = open(CONFIG_DIR).readlines()
all_content = [x.strip() for x in all_content if x[0] != "#"]
attacks = all_content[0].split(",")
attacks_target = all_content[1].split(",")
defenses = all_content[2].split(",")

# removing existing folders
for e_folder in ["sample_attacks", "sample_defenses", "sample_targeted_attacks"]:
    folder_dir = os.path.join(DEST, e_folder)
    try:
        shutil.rmtree(folder_dir)
    except:
        print(bcolors.WARNING + "Folder" + folder_dir + " have already been removed." + bcolors.ENDC)

# copy the whole folders into the destination
for e_folder in ["sample_attacks", "sample_defenses", "sample_targeted_attacks"]:
    folder_dir = os.path.join(DEST, e_folder)
    os.makedirs(folder_dir)
    for e_subfolder in os.listdir(e_folder):
        orig_folder = os.path.join(e_folder, e_subfolder)
        dest_folder = os.path.join(folder_dir, e_subfolder)
        if os.path.isfile(orig_folder):
            print(bcolors.OKBLUE + "Copy file:" + orig_folder + " to destination folder:" + dest_folder + bcolors.ENDC)
            shutil.copy2(orig_folder, dest_folder)
        elif e_subfolder in attacks + attacks_target + defenses:
            print(bcolors.OKBLUE + "Copy folder:" + orig_folder + " to destination folder:" + dest_folder + bcolors.ENDC)
            shutil.copytree(orig_folder, dest_folder)

# copy model and meta files into directory
for efile in os.listdir(META_DIR):
    efile_dir = os.path.join(META_DIR, efile)
    for e_folder in ["sample_attacks", "sample_targeted_attacks"]:
        for e_subfolder in os.listdir(os.path.join(DEST, e_folder)):
            if not os.path.isfile(e_subfolder) :
                dest_sub_dir = os.path.join(DEST, e_folder, e_subfolder)
                shutil.copy2(efile_dir, dest_sub_dir)

# and change file permissions
for e_folder in ["sample_attacks", "sample_targeted_attacks", "sample_defenses"]:
    for e_subfolder in os.listdir(os.path.join(DEST, e_folder)):
        dest_sub_dir = os.path.join(DEST, e_folder, e_subfolder)
        if not os.path.isfile(dest_sub_dir) :
            for mod_file in os.listdir(dest_sub_dir):
                if mod_file in ["run_defense.sh", "run_attack.sh"]:
                    mod_dir = os.path.join(dest_sub_dir, mod_file)
                    # this is only supported by python 3
                    print(bcolors.OKBLUE + "Change file mode for:"  + mod_dir + bcolors.ENDC)
                    os.chmod(mod_dir, 0o777)


# run the defense and attack
subprocess.call(['/home/ubuntu/cleverhans/examples/nips17_adversarial_competition/run_attacks_and_defenses.sh'])
