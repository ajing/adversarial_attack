import os
import shutil
import subprocess

DEST="/home/ubuntu/cleverhans/examples/nips17_adversarial_competition"
META_DIR = "/home/ubuntu/adversarial_attack/metafiles"



# removing existing folders
for e_folder in ["sample_attacks", "sample_defenses", "sample_targeted_attacks"]:
    folder_dir = os.path.join(DEST, e_folder)
    try:
        shutil.rmtree(folder_dir)
    except:
        print("Folder" + folder_dir + " have already been removed.")

# copy the whole folders into the destination
for e_folder in ["sample_attacks", "sample_defenses", "sample_targeted_attacks"]:
    folder_dir = os.path.join(DEST, e_folder)
    shutil.copytree(e_folder, folder_dir)

# copy model and meta files into directory
for efile in os.listdir(META_DIR):
    efile_dir = os.path.join(META_DIR, efile)
    for e_folder in ["sample_attacks", "sample_targeted_attacks"]:
        for e_subfolder in os.listdir(os.path.join(DEST, e_folder)):
            if not "." in e_subfolder:
                dest_sub_dir = os.path.join(DEST, e_folder, e_subfolder)
                shutil.copy2(efile_dir, dest_sub_dir)

# and change file permissions
for e_folder in ["sample_attacks", "sample_targeted_attacks", "sample_defenses"]:
    for e_subfolder in os.listdir(os.path.join(DEST, e_folder)):
        if not "." in e_subfolder:
            dest_sub_dir = os.path.join(DEST, e_folder, e_subfolder)
            for mod_file in os.listdir(dest_sub_dir):
                if mod_file in ["run_defense.sh", "run_attack.sh"]:
                    mod_dir = os.path.join(dest_sub_dir, mod_file)
                    # this is only supported by python 3
                    print("Change file mode for:"  + mod_dir)
                    os.chmod(mod_dir, 0o777)


# run the defense and attack
subprocess.call(['/home/ubuntu/cleverhans/examples/nips17_adversarial_competition/run_attacks_and_defenses.sh'])
