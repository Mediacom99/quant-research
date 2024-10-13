{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  # add binaries
  packages = with pkgs.python311Packages; [
    matplotlib
    openpyxl
    pandas
    numpy
    scipy
    scikit-learn
    seaborn

    #Text Editor stuff
    #Remove if you dont need it
    python-lsp-server
    pkgs.virtualenv
    
  ];

  # add build dependecies from list of derivations
  # inputsFrom = with pkgs; [
    
  # ];

#   # bash statements executed by nix-shell
#   shellHook = ''
#   export DEBUG=1;
# '';
}
