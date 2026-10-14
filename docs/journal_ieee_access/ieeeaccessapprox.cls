%% ieeeaccessapprox.cls
%% An approximation of the IEEE Access two-column journal layout, written because the
%% official ieeeaccess.cls was not available when this manuscript was prepared.
%% It mirrors the macro names of the official template (\history, \doi, \address,
%% \authorrefmark, \tfootnote, \markboth, \corresp, \headeretal, abstract, keywords,
%% \PARstart, biography) so that the manuscript body can be moved into the official
%% IEEE Access template by changing the \documentclass line only.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ieeeaccessapprox}[2026/10/13 approximation of the IEEE Access layout]
\LoadClass[10pt,twocolumn,letterpaper]{article}
\RequirePackage[letterpaper,top=0.95in,bottom=0.95in,left=0.62in,right=0.62in,columnsep=0.25in,headheight=15pt,headsep=0.22in,footskip=0.35in]{geometry}
\RequirePackage{mathptmx}
\RequirePackage[scaled=.92]{helvet}
\RequirePackage{microtype}
\RequirePackage{xcolor}
\RequirePackage{etoolbox}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage[font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false]{caption}
\captionsetup[table]{justification=centering,singlelinecheck=true,font=footnotesize}
\captionsetup[figure]{font=footnotesize}
\renewcommand\figurename{FIGURE}
\renewcommand\tablename{TABLE}
\renewcommand\thetable{\arabic{table}}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\renewcommand\baselinestretch{1}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 2pt}

% ---- section headings
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\Alph{subsection}}
\renewcommand\thesubsubsection{\arabic{subsubsection}}
\titleformat{\section}[block]{\normalfont\small\bfseries\centering\sffamily}{\thesection.}{0.5em}{\MakeUppercase}
\titleformat{name=\section,numberless}[block]{\normalfont\small\bfseries\centering\sffamily}{}{0pt}{\MakeUppercase}
\titleformat{\subsection}[block]{\normalfont\small\bfseries\itshape\sffamily}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}[runin]{\normalfont\small\itshape}{\thesubsubsection)}{0.4em}{}[.]
\titlespacing*{\section}{0pt}{11pt plus 2pt minus 2pt}{5pt plus 1pt}
\titlespacing*{\subsection}{0pt}{7pt plus 2pt minus 1pt}{3pt plus 1pt}
\titlespacing*{\subsubsection}{1em}{4pt plus 1pt}{0.6em}

% ---- metadata macros (names follow the official IEEE Access template)
\newcommand\history[1]{\gdef\@history{#1}}\gdef\@history{}
\newcommand\doi[1]{\gdef\@doi{#1}}\gdef\@doi{}
\newcommand\headeretal{\textit{et al.}}
\renewcommand\markboth[2]{\gdef\@runhead{#2}}\gdef\@runhead{}
\newcommand\corresp[1]{\gdef\@corresp{#1}}\gdef\@corresp{}
\newcommand\tfootnote[1]{\gdef\@tfootnote{#1}}\gdef\@tfootnote{}
\newcommand\authorrefmark[1]{\textsuperscript{#1}}
\gdef\@addrlist{}
\newcommand\address[2][]{\g@addto@macro\@addrlist{\par\textsuperscript{#1}#2}}
\newlength\titlepgskip
\titlepgskip=0pt

% ---- page styles
\pagestyle{fancy}
\fancyhf{}
\renewcommand\headrulewidth{0pt}
\fancyhead[C]{\sffamily\scriptsize\@runhead}
\fancyfoot[L]{\sffamily\scriptsize\bfseries VOLUME 14, 2026}
\fancyfoot[R]{\sffamily\scriptsize\bfseries\thepage}
\fancypagestyle{firstpage}{%
  \fancyhf{}\renewcommand\headrulewidth{0pt}
  \fancyhead[L]{\sffamily\scriptsize\parbox[t]{0.6\textwidth}{\@history\\ Digital Object Identifier \@doi}}
  \fancyfoot[L]{\sffamily\scriptsize\bfseries VOLUME 14, 2026}
  \fancyfoot[R]{\sffamily\scriptsize\bfseries\thepage}}

% ---- title block
\newcommand\@maketitleblock{%
  \begin{center}
    {\sffamily\bfseries\fontsize{21}{25}\selectfont \@title\par}
    \vskip 12pt
    {\sffamily\bfseries\fontsize{10}{12}\selectfont \@author\par}
    \vskip 4pt
    {\sffamily\fontsize{8}{9.5}\selectfont \@addrlist\par}
  \end{center}\vskip 6pt\vskip\titlepgskip}
\renewcommand\maketitle{\par
  \twocolumn[\@maketitleblock]%
  \thispagestyle{firstpage}%
  \begingroup\renewcommand\thefootnote{}%
  \ifdefempty{\@tfootnote}{}{\footnotetext{\sffamily\scriptsize\@tfootnote}}%
  \ifdefempty{\@corresp}{}{\footnotetext{\sffamily\scriptsize\@corresp}}%
  \endgroup}

% ---- abstract and keywords
\renewenvironment{abstract}{\par\noindent\bfseries\fontsize{9}{10.8}\selectfont\textsf{ABSTRACT}\hspace{0.6em}}{\par\vskip 5pt}
\newenvironment{keywords}{\par\noindent\bfseries\fontsize{9}{10.8}\selectfont\textsf{INDEX TERMS}\hspace{0.6em}}{\par\vskip 8pt}
\newcommand\PARstart[2]{\noindent{\large #1}\textsc{\MakeLowercase{#2}}}
\newenvironment{biography}[2]{\par\vskip 8pt\noindent{\sffamily\bfseries\small\MakeUppercase{#2}} }{\par}
\newcommand\EOD{}

% ---- bibliography
\renewenvironment{thebibliography}[1]
  {\section*{References}\footnotesize\sloppy
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth \advance\leftmargin\labelsep
         \itemsep 0pt \parsep 0pt \topsep 0pt
         \usecounter{enumiv}\let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy\clubpenalty4000\@clubpenalty\clubpenalty\widowpenalty4000}
  {\def\@noitemerr{\@latex@error{Empty `thebibliography' environment}\@ehc}\endlist}
\endinput
