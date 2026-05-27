// ===== FILE: contracts_-_bvm/utilities/BVMGiftVoucher.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

contract BVMGiftVoucher is BVMFeeBase {
    uint256 public mintFeeBvm = 100 ether;

    struct Voucher {
        address creator;
        uint128 valueWei;
        bytes32 redeemHash;
        bool    redeemed;
        address redeemer;
        uint64  mintedAt;
        uint64  redeemedAt;
    }
    Voucher[] public vouchers;
    mapping(address => uint256[]) public vouchersOf;

    event Minted(uint256 indexed id, address indexed creator, uint128 valueWei, bytes32 redeemHash);
    event Redeemed(uint256 indexed id, address indexed redeemer, uint128 valueWei);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error AlreadyRedeemed();
    error BadSecret();
    error EthSendFail();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function mint(bytes32 redeemHash) external payable returns (uint256 id) {
        if (msg.value == 0 || redeemHash == bytes32(0)) revert BadParams();
        _payFee(mintFeeBvm);
        id = vouchers.length;
        vouchers.push(Voucher({
            creator: msg.sender, valueWei: uint128(msg.value), redeemHash: redeemHash,
            redeemed: false, redeemer: address(0),
            mintedAt: uint64(block.timestamp), redeemedAt: 0
        }));
        vouchersOf[msg.sender].push(id);
        emit Minted(id, msg.sender, uint128(msg.value), redeemHash);
    }

    function redeem(uint256 id, bytes32 secret) external {
        Voucher storage v = vouchers[id];
        if (v.redeemed) revert AlreadyRedeemed();
        if (keccak256(abi.encodePacked(secret)) != v.redeemHash) revert BadSecret();
        v.redeemed = true; v.redeemer = msg.sender; v.redeemedAt = uint64(block.timestamp);
        (bool ok, ) = msg.sender.call{value: v.valueWei}("");
        if (!ok) revert EthSendFail();
        emit Redeemed(id, msg.sender, v.valueWei);
    }

    function totalVouchers() external view returns (uint256) { return vouchers.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(mintFeeBvm, next);
        mintFeeBvm = next;
    }
}


// ===== FILE: contracts_-_bvm/utilities/BVMFeeBase.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

interface IBVMTreasury {
    function collect(address payer, uint256 amount) external;
}

abstract contract BVMFeeBase {
    IBVMTreasury public immutable bvmTreasury;
    address      public owner;

    event FeeRouted(address indexed payer, uint256 amount, bytes4 indexed selector);
    event OwnershipTransferred(address indexed prev, address indexed next);

    error NotOwner();
    error ZeroAddress();

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }

    constructor(address _treasury, address _owner) {
        if (_treasury == address(0) || _owner == address(0)) revert ZeroAddress();
        bvmTreasury = IBVMTreasury(_treasury);
        owner       = _owner;
    }

    function _payFee(uint256 amount) internal {
        if (amount == 0) return;
        bvmTreasury.collect(msg.sender, amount);
        emit FeeRouted(msg.sender, amount, msg.sig);
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
