# Synwalk - Community Detection via Random Walk Modelling

Synwalk is a method for community detection in weighted networks with two-level hierarchies. This repository contains an integration of the Synwalk objective function into the [Infomap Software Package](https://github.com/mapequation/infomap) (see also see [www.mapequation.org](http://www.mapequation.org)), while preserving Infomap's original functionality. See [here](https://github.com/synwalk/synwalk-analysis) for an evaluation framework of our method.


Getting started:
--------------------------------------------------------
In a terminal with the GNU Compiler Collection installed, just run `make` in the current directory to compile the,code with the included `Makefile`. Call `./Infomap --synwalk` to run Synwalk (omit the `--synwalk` argument to run Infomap instead). Run `./Infomap --help` for a list of available options.


Authors:
--------------------------------------------------------
[Christian Toth](https://github.com/chritoth).

See http://www.mapequation.org/about.html for information about Daniel Edler, and Martin Rosvall, who are the authors of the original package.


Terms of use:
--------------------------------------------------------
The Infomap software is released under the GNU Affero General Public License version 3
or any later version (see LICENSE_AGPLv3.txt.), which we retain for this derivative work.