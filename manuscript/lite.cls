\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lite}[Lite template class]
\LoadClass[12pt]{article}

% packages
\RequirePackage[
    pdfencoding=auto,
    % colorlinks=true,
    % linkcolor=blue,
    % citecolor=blue,
    % urlcolor=blue,
]{hyperref}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{amsmath,amssymb,mathtools}
% \RequirePackage{newtxtext,newtxmath}
\RequirePackage{bookmark}
\RequirePackage{times}           % Times New Roman (closest approx.)
\RequirePackage{caption}         % go to the top of an image
\RequirePackage{anyfontsize}     % select any font size
\RequirePackage{dirtytalk}       % proper quotation marks
\RequirePackage{parskip}         % skip paragraph indent
\RequirePackage{xcolor}          % used for custom colors
\RequirePackage{pifont}          % used for xmark/cmark
\RequirePackage{lipsum}          % generate dummy text
\RequirePackage{silence}         % silence warnings
\RequirePackage{textcomp}        % registered, copyright, etc
\RequirePackage{moresize}        % more font size commands
\RequirePackage{titlesec}        % customize \section etc
\RequirePackage{enumitem}        % customize itemize
\RequirePackage{xspace}          % \xspace: used in defining new cmds
\RequirePackage{empheq}          % put box around equations
\RequirePackage{bm}              % math bold font

% margin line numbers for review drafts; inert unless \linenumbers is issued
\RequirePackage[mathlines]{lineno}
% lineno drops out inside amsmath display environments unless they are wrapped
\newcommand*\patchAmsMathEnvironmentForLineno[1]{%
    \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
    \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
    \renewenvironment{#1}%
        {\linenomath\csname old#1\endcsname}%
        {\csname oldend#1\endcsname\endlinenomath}}
\newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
    \patchAmsMathEnvironmentForLineno{#1}%
    \patchAmsMathEnvironmentForLineno{#1*}}
\AtBeginDocument{%
    \patchBothAmsMathEnvironmentsForLineno{equation}%
    \patchBothAmsMathEnvironmentsForLineno{align}%
    \patchBothAmsMathEnvironmentsForLineno{flalign}%
    \patchBothAmsMathEnvironmentsForLineno{alignat}%
    \patchBothAmsMathEnvironmentsForLineno{gather}%
    \patchBothAmsMathEnvironmentsForLineno{multline}%
}

% table related packages
\RequirePackage{booktabs}        % pro tables (from NeurIPS)
\RequirePackage{tabularx}        % Used for custom width tables
\RequirePackage{array}           % Required for customizing X columns
\newcolumntype{C}{>{\centering\arraybackslash}X}

% Use clever ref (cref) and edefine the label format for figures
\RequirePackage{cleveref}
\crefname{figure}{Fig.}{Figs.}
\Crefname{figure}{Figure}{Figures}

% embed images
\RequirePackage{float}
\RequirePackage{wrapfig}
\RequirePackage{etoc}
\RequirePackage{graphicx}
\graphicspath{{./figs/}}

% custom sectioning
\titleformat*{\section}{\Large\bfseries\sffamily}
\titleformat*{\subsection}{\large\bfseries\sffamily}
\titleformat*{\subsubsection}{\bfseries\sffamily}
\setcounter{secnumdepth}{-2}  % no numbering for sections

% define colors
\definecolor{neuro}{HTML}{39A300}
\definecolor{comp}{HTML}{077AF8}
\definecolor{theo}{HTML}{EF4136}
\definecolor{RowColorTop}{HTML}{e0dccb}
\definecolor{RowColorDark}{HTML}{f2efe6}
\definecolor{RowColorLight}{HTML}{fdfbf7}
\definecolor{MSBlue}{rgb}{.204,.353,.541}
\definecolor{MSLightBlue}{rgb}{.31,.506,.741}

% new commands
\newcommand{\xmark}{{\Large\color{red}\ding{55}}}
\newcommand{\cmark}{{\Large\color{green}\ding{51}}}
\def\todo#1{{\color{red}$\mathtt{TODO:}$ {#1}}}
\def\attn#1{{\color{purple}$\left<\mathtt{attn!}\right>$ {#1} $\left</\mathtt{attn!}\right>$}}
\def\sayit#1{\say{\textit{{#1}}}}
\def\quote#1#2{\hspace{0.1\textwidth}\begin{tabular}{|p{0.80\textwidth}}\say{{#1}} --- {#2}\end{tabular}}

% declare math operators
\DeclareMathOperator*{\maximize}{maximize_{\mathnormal{}}}
\DeclareMathOperator*{\minimize}{minimize_{\mathnormal{}}}
\DeclareMathOperator*{\argmax}{argmax_{\mathnormal{}}}
\DeclareMathOperator*{\argmin}{argmin_{\mathnormal{}}}
\DeclareMathOperator{\EE}{\mathbb{E}}
\DeclareMathOperator{\RR}{\mathbb{R}}
\DeclareMathOperator{\ZZ}{\mathbb{Z}}
\DeclareMathOperator{\BB}{\mathcal{B}}
\DeclareMathOperator{\CC}{\mathcal{C}}
\DeclareMathOperator{\FF}{\mathcal{F}}
\DeclareMathOperator{\GG}{\mathcal{G}}
\DeclareMathOperator{\HH}{\mathcal{H}}
\DeclareMathOperator{\LL}{\mathcal{L}}
\DeclareMathOperator{\MM}{\mathcal{M}}
\DeclareMathOperator{\NN}{\mathcal{N}}
\DeclareMathOperator{\PP}{\mathcal{P}}

% declare robust math commands
\DeclareRobustCommand{\xx}{\bm{x}}
\DeclareRobustCommand{\zz}{\bm{z}}
\DeclareRobustCommand{\rr}{\bm{r}}
\DeclareRobustCommand{\dr}{\bm{\delta r}}

% new math commands
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\kl}[2]{\mathcal{D}_{\mathtt{KL}}\left({#1}\,\|\,{#2}\right)}
\newcommand{\KL}[2]{\mathcal{D}_{\mathtt{KL}}\Big{(}{#1}\,\big\|\,{#2}\Big{)}}
\newcommand{\expec}[2]{\EE_{#1} [ {#2} ]}
\newcommand{\expect}[2]{\EE_{#1} \Big{[} {#2} \Big{]}}
\newcommand{\EXPECT}[2]{\EE_{#1} \Bigg{[} {#2} \Bigg{]}}

% p-vae related commands
\newcommand{\pois}{\mathcal{P}\mathrm{ois}}
\newcommand{\pvae}{$\PP$-VAE\xspace}

% Ca related commands
\def\SP#1{\textsuperscript{{#1}}}
\def\SB#1{\textsubscript{{#1}}}
\def\SPSB#1#2{\rlap{\SP{#1}}\SB{#2}}
\newcommand{\CAS}{Ca\SPSB{2+}{\it slow} }
\newcommand{\CAF}{Ca\SPSB{2+}{\it fast} }
\newcommand{\CA}{Ca\texorpdfstring{\textsuperscript{2+}}{\texttwoinferior} }

\newcommand\titlename[3]{
    \begin{center}
        \textbf{\LARGE\sffamily\color{MSBlue} {#1}} \\
        \vspace{0.7em}
        {\fontsize{16}{18}\selectfont {#2}} \\
        {\small {#3}}
    \end{center}
    \noindent\rule[1.5ex]{\textwidth}{0.8pt}
}
