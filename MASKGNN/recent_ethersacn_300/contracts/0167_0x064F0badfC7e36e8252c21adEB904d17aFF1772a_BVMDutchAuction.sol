// ===== FILE: contracts_-_bvm/utilities/BVMDutchAuction.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import "./BVMFeeBase.sol";

interface IERC20Min { function transfer(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }

contract BVMDutchAuction is BVMFeeBase {
    uint256 public listFeeBvm = 210 ether;

    struct Auction {
        address seller;
        address token;
        uint128 amount;
        uint128 startPriceWei;
        uint128 floorPriceWei;
        uint64  startsAt;
        uint64  endsAt;
        bool    settled;
        address buyer;
        uint128 settlePriceWei;
    }
    Auction[] public auctions;
    mapping(address => uint256[]) public auctionsBy;

    event Listed(uint256 indexed id, address indexed seller, address indexed token, uint128 amount, uint128 startPrice, uint128 floorPrice, uint64 startsAt, uint64 endsAt);
    event Settled(uint256 indexed id, address indexed buyer, uint128 pricePaid);
    event Cancelled(uint256 indexed id);
    event FeeChanged(uint256 prev, uint256 next);

    error BadParams();
    error NotSeller();
    error AlreadySettled();
    error NotStarted();
    error Expired();
    error PaymentShort();
    error EthSendFail();
    error TransferFailed();

    constructor(address _treasury, address _owner) BVMFeeBase(_treasury, _owner) {}

    function list(address token, uint128 amount, uint128 startPriceWei, uint128 floorPriceWei, uint64 startsAt, uint64 endsAt)
        external returns (uint256 id)
    {
        if (token == address(0) || amount == 0 || startPriceWei < floorPriceWei || endsAt <= startsAt || startsAt < block.timestamp) revert BadParams();
        _payFee(listFeeBvm);
        if (!IERC20Min(token).transferFrom(msg.sender, address(this), amount)) revert TransferFailed();
        id = auctions.length;
        auctions.push();
        Auction storage a = auctions[id];
        a.seller = msg.sender; a.token = token; a.amount = amount;
        a.startPriceWei = startPriceWei; a.floorPriceWei = floorPriceWei;
        a.startsAt = startsAt; a.endsAt = endsAt;
        auctionsBy[msg.sender].push(id);
        emit Listed(id, msg.sender, token, amount, startPriceWei, floorPriceWei, startsAt, endsAt);
    }

    function priceAt(uint256 id) public view returns (uint128) {
        Auction storage a = auctions[id];
        if (block.timestamp <= uint256(a.startsAt)) return a.startPriceWei;
        if (block.timestamp >= uint256(a.endsAt))   return a.floorPriceWei;
        uint256 elapsed  = block.timestamp - uint256(a.startsAt);
        uint256 duration = uint256(a.endsAt) - uint256(a.startsAt);
        uint256 drop     = (uint256(a.startPriceWei) - uint256(a.floorPriceWei)) * elapsed / duration;
        return uint128(uint256(a.startPriceWei) - drop);
    }

    function buy(uint256 id) external payable {
        Auction storage a = auctions[id];
        if (a.settled) revert AlreadySettled();
        if (block.timestamp < uint256(a.startsAt)) revert NotStarted();
        if (block.timestamp > uint256(a.endsAt))   revert Expired();
        uint128 price = priceAt(id);
        if (msg.value < uint256(price)) revert PaymentShort();
        a.settled = true; a.buyer = msg.sender; a.settlePriceWei = price;
        (bool ok, ) = a.seller.call{value: price}("");
        if (!ok) revert EthSendFail();
        if (msg.value > uint256(price)) {
            (bool ok2, ) = msg.sender.call{value: msg.value - price}("");
            if (!ok2) revert EthSendFail();
        }
        if (!IERC20Min(a.token).transfer(msg.sender, a.amount)) revert TransferFailed();
        emit Settled(id, msg.sender, price);
    }

    function cancel(uint256 id) external {
        Auction storage a = auctions[id];
        if (msg.sender != a.seller) revert NotSeller();
        if (a.settled) revert AlreadySettled();
        if (block.timestamp <= uint256(a.endsAt)) revert NotStarted();
        a.settled = true;
        if (!IERC20Min(a.token).transfer(a.seller, a.amount)) revert TransferFailed();
        emit Cancelled(id);
    }

    function totalAuctions() external view returns (uint256) { return auctions.length; }

    function setFee(uint256 next) external onlyOwner {
        emit FeeChanged(listFeeBvm, next);
        listFeeBvm = next;
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
