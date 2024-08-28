with import <nixpkgs> {};

let
  unstable = import
    (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz")
    # reuse the current configuration
    { config = config; };
in
pkgs.mkShell {

  # set derivation name
  name = "quant-start";
  
  # add binaries
  packages = with pkgs.python311Packages; [
    matplotlib
    openpyxl
    pandas
    numpy
    scipy
    scikit-learn
    seaborn
  ];

  # add build dependecies from list of derivations
  # inputsFrom = with pkgs; [
    
  # ];

#   # bash statements executed by nix-shell
#   shellHook = ''
#   export DEBUG=1;
# '';
}
