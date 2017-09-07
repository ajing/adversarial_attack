DEST="/home/ubuntu/cleverhans/examples/nips17_adversarial_competition"

# add necessary model files into existing attacks

# change all file permissions

# removing existing folders
rm -r $DEST/sample_attacks
rm -r $DEST/sample_defenses
rm -r $DEST/sample_targeted_attacks
# copy folders over
cp -r sample_attacks $DEST
cp -r sample_defenses $DEST
cp -r sample_targeted_attacks $DEST
