// ===== FILE: Daemons/DaemonToken.sol =====
// SPDX-License-Identifier: MIT
/*
Mutating onchain artifacts powered by uniswap v4
DAEMONS is a generative onchain art project built on top of the ERC20 object model: 
every whole token corresponds to a unique work of databent glitch art PFPs.

Website: https://daemons.life/
X: https://x.com/DAEMONSv4
*/
pragma solidity 0.8.34;

interface IHookCtl {
    function start() external;
}

interface IDaemonHook {
    function corruption(bytes32 poolId) external view returns (bytes32);
    function defaultPoolId() external view returns (bytes32);
    function spawnBlockByPool(bytes32 poolId) external view returns (uint64);
}

interface IDaemonRenderer {
    function objectURI(address holder, uint256 slot, bytes32 corruption, uint64 spawnBlock) external view returns (string memory);
}

contract DaemonToken {
    string public constant name     = "DAEMONS";
    string public constant symbol   = "DAEMON";
    uint8  public constant decimals = 18;

    uint256 public constant SUPPLY = 6_666 * 1e18;
    uint256 public totalSupply = SUPPLY;

    address public constant DEAD = 0x000000000000000000000000000000000000dEaD;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => bool) public exempt;
    mapping(address => uint64) public firstHeldAt;

    address public admin;
    address public hook;
    address public renderer;
    address public wrapper;

    bool   public live;
    uint64 public livePulseBlock;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed holder, address indexed spender, uint256 value);
    event Burned(address indexed from, uint256 amount);
    event LivePulse(uint256 atBlock, uint64 atTimestamp);
    event ExemptSet(address indexed account, bool flag);
    event HookSet(address indexed prev, address indexed next);
    event RendererSet(address indexed prev, address indexed next);
    event WrapperSet(address indexed prev, address indexed next);
    event AdminTransferred(address indexed prev, address indexed next);

    error InsufficientBalance();
    error InsufficientAllowance();
    error ZeroAddress();
    error NotAdmin();
    error NotLive();
    error AlreadyLive();

    modifier onlyAdmin() { if (msg.sender != admin) revert NotAdmin(); _; }

    constructor() {
        admin = msg.sender;
        balanceOf[msg.sender] = SUPPLY;
        exempt[msg.sender] = true;
        exempt[DEAD]       = true;
        firstHeldAt[msg.sender] = uint64(block.number);
        emit Transfer(address(0), msg.sender, SUPPLY);
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _move(msg.sender, to, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        uint256 a = allowance[from][msg.sender];
        if (a != type(uint256).max) {
            if (a < value) revert InsufficientAllowance();
            unchecked { allowance[from][msg.sender] = a - value; }
        }
        _move(from, to, value);
        return true;
    }

    function _move(address from, address to, uint256 value) internal {
        if (to == address(0)) revert ZeroAddress();
        if (balanceOf[from] < value) revert InsufficientBalance();
        if (!live && !exempt[from] && !exempt[to]) revert NotLive();
        if (balanceOf[to] == 0 && firstHeldAt[to] == 0 && to != DEAD) {
            firstHeldAt[to] = uint64(block.number);
        }
        unchecked {
            balanceOf[from] -= value;
            balanceOf[to]   += value;
        }
        emit Transfer(from, to, value);
    }

    function burn(uint256 value) external {
        if (balanceOf[msg.sender] < value) revert InsufficientBalance();
        unchecked {
            balanceOf[msg.sender] -= value;
            totalSupply -= value;
        }
        emit Transfer(msg.sender, address(0), value);
        emit Burned(msg.sender, value);
    }

    function holdings(address holder) external view returns (uint256 wholeArtifacts) {
        return balanceOf[holder] / 1e18;
    }

    function seedFor(address holder, uint256 slot) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(holder, slot, "DAEMON"));
    }

    function objectURI(address holder, uint256 slot) external view returns (string memory) {
        if (renderer == address(0)) return "";
        bytes32 corr;
        uint64  spawn;
        if (hook != address(0)) {
            bytes32 poolId = IDaemonHook(hook).defaultPoolId();
            corr  = IDaemonHook(hook).corruption(poolId);
            spawn = IDaemonHook(hook).spawnBlockByPool(poolId);
        }
        return IDaemonRenderer(renderer).objectURI(holder, slot, corr, spawn);
    }

    function setHook(address h) external onlyAdmin {
        emit HookSet(hook, h);
        hook = h;
        if (h != address(0)) {
            exempt[h] = true;
            emit ExemptSet(h, true);
        }
    }

    function setRenderer(address r) external onlyAdmin {
        emit RendererSet(renderer, r);
        renderer = r;
    }

    function setWrapper(address w) external onlyAdmin {
        emit WrapperSet(wrapper, w);
        wrapper = w;
        if (w != address(0)) {
            exempt[w] = true;
            emit ExemptSet(w, true);
        }
    }

    function setExempt(address account, bool flag) external onlyAdmin {
        if (account == address(0)) revert ZeroAddress();
        exempt[account] = flag;
        emit ExemptSet(account, flag);
    }

    function setManyExempt(address[] calldata accounts, bool flag) external onlyAdmin {
        for (uint256 i; i < accounts.length; ) {
            address a = accounts[i];
            if (a != address(0)) {
                exempt[a] = flag;
                emit ExemptSet(a, flag);
            }
            unchecked { i++; }
        }
    }

    function goLive() external onlyAdmin {
        if (live) revert AlreadyLive();
        live           = true;
        livePulseBlock = uint64(block.number);
        if (hook != address(0)) {
            try IHookCtl(hook).start() {} catch {}
        }
        emit LivePulse(block.number, uint64(block.timestamp));
    }

    function transferAdmin(address next) external onlyAdmin {
        if (next == address(0)) revert ZeroAddress();
        emit AdminTransferred(admin, next);
        admin = next;
    }

    function renounceAdmin() external onlyAdmin {
        emit AdminTransferred(admin, address(0));
        admin = address(0);
    }
}
