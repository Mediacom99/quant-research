# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi


# The following lines were added by compinstall
zstyle :compinstall filename '/home/edoardo/.zshrc'

autoload -Uz compinit
fpath+=~/.zfunc
compinit
# End of lines added by compinstall
# Lines configured by zsh-newuser-install
HISTFILE=~/.histfile
HISTSIZE=10000
SAVEHIST=10000
setopt extendedglob
unsetopt beep nomatch
bindkey -e
# End of lines configured by zsh-newuser-install

# Personal configuration

alias grep='grep --color=auto'
alias ls='ls --color=auto'
alias ll='ls -lah --color=auto'
alias la='ls -ah --color=auto'
alias poweroff='sudo systemctl poweroff'
alias reboot='sudo systemctl reboot'
alias python='/usr/bin/python'
alias du='du --si'

export XDG_CONFIG_HOME="${HOME}/.config"
export TERM=alacritty
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin:$HOME/zig/zig-master


# Generated for envman. Do not edit.
[ -s "$HOME/.config/envman/load.sh" ] && source "$HOME/.config/envman/load.sh"
source ~/powerlevel10k/powerlevel10k.zsh-theme

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
