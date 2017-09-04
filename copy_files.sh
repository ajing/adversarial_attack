DEST="/home/ubuntu/cleverhans/examples/nips17_adversarial_competition"
# removing existing folders
rm $DEST/sample_attacks
rm $DEST/sample_defenses
rm $DEST/sample_targeted_attacks
# copy folders over
cp sample_attacks $DEST
cp sample_defenses $DEST
cp sample_targeted_attacks $DEST
