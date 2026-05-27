// ===== FILE: contracts_-_bvm/BVMTreasury.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IBVM {
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function balanceOf(address) external view returns (uint256);
}

interface INodes {
    function applyTribute(uint256 amount) external;
    function totalUnits() external view returns (uint256);
}

contract BVMTreasury {
    IBVM    public immutable token;
    INodes  public nodes;
    address public owner;

    mapping(address => bool) public authorized;

    uint256 public totalPaid;
    uint256 public totalCollected;

    event NodesBound(address indexed nodes);
    event Paid(address indexed to, uint256 amount);
    event Collected(address indexed utility, address indexed payer, uint256 amount);
    event AuthorizedSet(address indexed who, bool flag);
    event OwnershipTransferred(address indexed prev, address indexed next);

    error NotOwner();
    error NotNodes();
    error NotAuthorized();
    error AlreadyBound();
    error ZeroAddress();
    error ZeroAmount();
    error TransferFailed();

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }
    modifier onlyNodes() { if (msg.sender != address(nodes)) revert NotNodes(); _; }

    constructor(address _token) {
        if (_token == address(0)) revert ZeroAddress();
        token = IBVM(_token);
        owner = msg.sender;
    }

    function bindNodes(address n) external onlyOwner {
        if (n == address(0)) revert ZeroAddress();
        if (address(nodes) != address(0)) revert AlreadyBound();
        nodes = INodes(n);
        emit NodesBound(n);
    }

    function setAuthorized(address who, bool flag) external onlyOwner {
        if (who == address(0)) revert ZeroAddress();
        authorized[who] = flag;
        emit AuthorizedSet(who, flag);
    }

    function setAuthorizedMany(address[] calldata who, bool flag) external onlyOwner {
        for (uint256 i; i < who.length; ) {
            address a = who[i];
            if (a != address(0)) {
                authorized[a] = flag;
                emit AuthorizedSet(a, flag);
            }
            unchecked { i++; }
        }
    }

    function reserveLeft() external view returns (uint256) {
        return token.balanceOf(address(this));
    }

    function payOut(address to, uint256 amount) external onlyNodes returns (uint256 paid) {
        if (to == address(0)) revert ZeroAddress();
        uint256 bal = token.balanceOf(address(this));
        if (bal == 0 || amount == 0) return 0;
        paid = amount > bal ? bal : amount;
        if (!token.transfer(to, paid)) revert TransferFailed();
        unchecked { totalPaid += paid; }
        emit Paid(to, paid);
    }

    function collect(address payer, uint256 amount) external {
        if (!authorized[msg.sender]) revert NotAuthorized();
        if (amount == 0) revert ZeroAmount();
        if (!token.transferFrom(payer, address(this), amount)) revert TransferFailed();
        if (address(nodes) != address(0)) {
            uint256 units = nodes.totalUnits();
            if (units > 0) nodes.applyTribute(amount);
        }
        unchecked { totalCollected += amount; }
        emit Collected(msg.sender, payer, amount);
    }

    function transferOwnership(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, next);
        owner = next;
    }

    function renounceOwnership() external onlyOwner {
        emit OwnershipTransferred(owner, address(0));
        owner = address(0);
    }
}
