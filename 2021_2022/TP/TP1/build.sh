for d in Poisson*
do
	cd $d
	python build.py
	cd -
done

