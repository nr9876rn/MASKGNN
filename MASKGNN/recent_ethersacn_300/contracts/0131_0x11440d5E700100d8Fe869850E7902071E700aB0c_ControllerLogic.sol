// ===== FILE: contracts/core/ControllerLogic.sol =====
/**
 * SPDX-License-Identifier: UNLICENSED
 */
pragma solidity =0.6.10;

pragma experimental ABIEncoderV2;

import {OwnableUpgradeSafe} from "../packages/oz/upgradeability/OwnableUpgradeSafe.sol";
import {ReentrancyGuardUpgradeSafe} from "../packages/oz/upgradeability/ReentrancyGuardUpgradeSafe.sol";
import {Initializable} from "../packages/oz/upgradeability/Initializable.sol";
import {SafeMath} from "../packages/oz/SafeMath.sol";
import {MarginVault} from "../libs/MarginVault.sol";
import {Actions} from "../libs/Actions.sol";
import {ERC20Interface} from "../interfaces/ERC20Interface.sol";
import {AddressBookInterface} from "../interfaces/AddressBookInterface.sol";
import {OtokenInterface} from "../interfaces/OtokenInterface.sol";
import {MarginCalculatorInterface} from "../interfaces/MarginCalculatorInterface.sol";
import {OracleInterface} from "../interfaces/OracleInterface.sol";
import {WhitelistInterface} from "../interfaces/WhitelistInterface.sol";
import {MarginPoolInterface} from "../interfaces/MarginPoolInterface.sol";
import {ControllerInterface} from "../interfaces/ControllerInterface.sol";

/**
 * @title Settlement
 * @author Rysk Finance
 * @notice Contract that handles logic for the Controller on a modified version of Opyn's Gamma protocol
 *         An extension of the Controller to avoid contract size issues. Error list can be found in Controller.sol
 *         Main functions callable by Controller
 */
contract ControllerLogic is Initializable, OwnableUpgradeSafe, ReentrancyGuardUpgradeSafe {
    using MarginVault for MarginVault.Vault;
    using SafeMath for uint256;

    AddressBookInterface public addressbook;
    WhitelistInterface public whitelist;
    MarginCalculatorInterface public calculator;
    OracleInterface public oracle;
    ControllerInterface public controller;
    MarginPoolInterface public pool;

    ///@dev scale used in MarginCalculator
    uint256 internal constant BASE = 8;

    ///@dev the number of seconds an ITM option is redeemable for after its expiry before the vault is settleable and collateral claimable.
    uint256 public redeemTimePeriod;

    /// @notice emits an event when a long oToken is deposited into a vault
    event LongOtokenDeposited(
        address indexed otoken,
        address indexed accountOwner,
        address indexed from,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a long oToken is withdrawn from a vault
    event LongOtokenWithdrawed(
        address indexed otoken,
        address indexed AccountOwner,
        address indexed to,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a collateral asset is deposited into a vault
    event CollateralAssetDeposited(
        address indexed asset,
        address indexed accountOwner,
        address indexed from,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a collateral asset is withdrawn from a vault
    event CollateralAssetWithdrawed(
        address indexed asset,
        address indexed AccountOwner,
        address indexed to,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a short oToken is minted from a vault
    event ShortOtokenMinted(
        address indexed otoken,
        address indexed AccountOwner,
        address indexed to,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a short oToken is burned
    event ShortOtokenBurned(
        address indexed otoken,
        address indexed AccountOwner,
        address indexed from,
        uint256 vaultId,
        uint256 amount
    );
    /// @notice emits an event when a vault is settled
    event VaultSettled(
        address indexed accountOwner,
        address indexed oTokenAddress,
        address to,
        uint256 collateralPayout,
        uint256 strikePayout,
        uint256 vaultId,
        uint256 indexed vaultType
    );
    /// @notice emits an event when an oToken is redeemed
    event Redeem(
        address indexed otoken,
        address indexed redeemer,
        address indexed receiver,
        address collateralAsset,
        uint256 otokenBurned,
        uint256 payout
    );

    /// @notice emits an event when the redeem time period is updated
    event RedeemTimePeriodUpdated(uint256 previousPeriod, uint256 newPeriod);

    struct SettleMem {
        address collateral;
        address underlying;
        address strike;
        uint256 expiry;
        uint256 collateralPayout;
        uint256 strikePayout;
        uint256 strikeCount;
        uint256 shortAmount;
        bool isValidVault;
        uint256 collateralRedemptionBalance;
        uint256 receivingAssetBalance;
        uint256 otokenQuantity;
        uint256 underlyingExpiryPrice;
        uint256 collateralExpiryPrice;
    }

    /**
     * @notice check if the sender is the Controller module
     */
    modifier onlyController() {
        require(
            msg.sender == AddressBookInterface(addressbook).getController(),
            "ControllerLogic: Sender is not Controller"
        );

        _;
    }

    /**
     * @notice initalize the deployed contract
     * @param _addressbook addressbook module
     * @param _owner account owner address
     */
    function initialize(address _addressbook, address _owner) external initializer {
        require(_addressbook != address(0), "C7");
        require(_owner != address(0), "C8");

        __Ownable_init(_owner);
        __ReentrancyGuard_init_unchained();
        addressbook = AddressBookInterface(_addressbook);
        redeemTimePeriod = 3600;
    }

    /**
     * @dev updates the configuration of the controller. can only be called by the owner
     */
    function refreshConfiguration() external onlyOwner {
        _refreshConfigInternal();
    }

    /**
     * @notice get the collateral amount a redeemer receives for a given quantity of expired oTokens
     * @dev for cash-settled options: this is the cash value of the options in collateral terms.
     * for physically settled options: this is the collateral delivery amount (what the holder receives
     * in exchange for their strike payment during handleRedeem).
     * @param _otoken oToken address
     * @param _amount amount of oTokens to calculate the payout for (1e8 scaled)
     * @return amount of collateral to transfer to the redeemer
     */
    function getPayout(address _otoken, uint256 _amount) public view returns (uint256) {
        return calculator.getExpiredPayoutRate(_otoken).mul(_amount).div(10**BASE);
    }

    /**
     * @notice set ITM option redemption time period
     * @dev can only be called by owner
     * @param _redeemTimePeriod number of seconds an ITM option is redeemable for after its expiry
     */
    function setRedeemTimePeriod(uint256 _redeemTimePeriod) external onlyOwner {
        require(_redeemTimePeriod > 0, "C38");

        emit RedeemTimePeriodUpdated(redeemTimePeriod, _redeemTimePeriod);
        redeemTimePeriod = _redeemTimePeriod;
    }

    function handleDepositLong(Actions.DepositArgs memory _args) external onlyController {
        require(whitelist.isWhitelistedOtoken(_args.asset), "C17");

        OtokenInterface otoken = OtokenInterface(_args.asset);

        require(now < otoken.expiryTimestamp(), "C18");
        // dont allow deposit longs for physically settled type 2 vaults and physically settled otokens
        require(!otoken.isPhysicallySettled(), "C43");
        (, uint256 typeVault, ) = controller.getVaultWithDetails(_args.owner, _args.vaultId);
        require(typeVault != 2, "C43");

        controller.updateVault(2, _args.owner, _args.vaultId, _args.asset, _args.amount);

        pool.transferToPool(_args.asset, _args.from, _args.amount);

        emit LongOtokenDeposited(_args.asset, _args.owner, _args.from, _args.vaultId, _args.amount);
    }

    function handleWithdrawLong(Actions.WithdrawArgs memory _args) external onlyController {
        OtokenInterface otoken = OtokenInterface(_args.asset);

        require(now < otoken.expiryTimestamp(), "C19");
        controller.updateVault(3, _args.owner, _args.vaultId, _args.asset, _args.amount);

        pool.transferToUser(_args.asset, _args.to, _args.amount);

        emit LongOtokenWithdrawed(_args.asset, _args.owner, _args.to, _args.vaultId, _args.amount);
    }

    function handleDepositCollateral(Actions.DepositArgs memory _args) external onlyController {
        require(whitelist.isWhitelistedCollateral(_args.asset), "C21");

        controller.updateVault(4, _args.owner, _args.vaultId, _args.asset, _args.amount);

        pool.transferToPool(_args.asset, _args.from, _args.amount);

        emit CollateralAssetDeposited(_args.asset, _args.owner, _args.from, _args.vaultId, _args.amount);
    }

    function handleWithdrawCollateral(Actions.WithdrawArgs memory _args) external onlyController {
        (MarginVault.Vault memory vault, , ) = controller.getVaultWithDetails(
            _args.owner,
            _args.vaultId
        );

        if (_isNotEmpty(vault.shortOtokens)) {
            OtokenInterface otoken = OtokenInterface(vault.shortOtokens[0]);

            require(now < otoken.expiryTimestamp(), "C22");
        }

        controller.updateVault(5, _args.owner, _args.vaultId, _args.asset, _args.amount);

        pool.transferToUser(_args.asset, _args.to, _args.amount);

        emit CollateralAssetWithdrawed(_args.asset, _args.owner, _args.to, _args.vaultId, _args.amount);
    }

    function handleMintOtoken(Actions.MintArgs memory _args, uint256 vaultType) external onlyController {
        require(whitelist.isWhitelistedOtoken(_args.otoken), "C23");

        OtokenInterface otoken = OtokenInterface(_args.otoken);

        require(now < otoken.expiryTimestamp(), "C24");

        if (vaultType == 2) {
            // this is a physically settled vault
            require(otoken.isPhysicallySettled(), "C40");

            uint256 collateralDecimals = uint256(ERC20Interface(otoken.collateralAsset()).decimals());

            // prevent truncation in collateral tracking for tokens with fewer decimals than oTokens (8)
            if (collateralDecimals < 8) {
                require(_args.amount % (10**(8 - collateralDecimals)) == 0, "C46");
            }

            // update otokenQuantity and collateralBalance using canonical-balance approach:
            // compute collateralBalance from total otokenQuantity before and after, so the result is
            // independent of how many mints occur. this prevents rounding discrepancies between
            // split mints and combined burns/redeems.
            (, , uint256 oldQuantity) = pool.getRedemptionBalance(_args.otoken);
            uint256 newQuantity = oldQuantity.add(_args.amount);

            if(otoken.isPut()) {
                uint256 oldCollateral = oldQuantity.mul(otoken.strikePrice()).mul(10**collateralDecimals).div(1e16);
                uint256 newCollateral = newQuantity.mul(otoken.strikePrice()).mul(10**collateralDecimals).div(1e16);
                pool.updateRedemptionBalance(
                    _args.otoken,
                    int256(_args.amount),
                    false,
                    int256(newCollateral.sub(oldCollateral))
                );
            } else {
                uint256 oldCollateral = oldQuantity.mul(10**collateralDecimals).div(1e8);
                uint256 newCollateral = newQuantity.mul(10**collateralDecimals).div(1e8);
                pool.updateRedemptionBalance(
                    _args.otoken,
                    int256(_args.amount),
                    false,
                    int256(newCollateral.sub(oldCollateral))
                );
            }
        } else {
            // this is not a physically settled vault
            require(!otoken.isPhysicallySettled(), "C40");
        }

        controller.updateVault(0, _args.owner, _args.vaultId, _args.otoken, _args.amount);

        otoken.mintOtoken(_args.to, _args.amount);

        emit ShortOtokenMinted(_args.otoken, _args.owner, _args.to, _args.vaultId, _args.amount);
    }

    function handleBurnOtoken(Actions.BurnArgs memory _args, uint256 vaultType) external onlyController {
        OtokenInterface otoken = OtokenInterface(_args.otoken);

        // do not allow burning expired otoken
        require(now < otoken.expiryTimestamp(), "C26");

        if (vaultType == 2) {
            // this is a physically settled vault
            require(otoken.isPhysicallySettled(), "C40");

            uint256 collateralDecimals = uint256(ERC20Interface(otoken.collateralAsset()).decimals());

            // prevent truncation in collateral tracking for tokens with fewer decimals than oTokens (8)
            if (collateralDecimals < 8) {
                require(_args.amount % (10**(8 - collateralDecimals)) == 0, "C46");
            }

            // canonical-balance approach for burns (mirrors mint logic above)
            (, , uint256 oldQuantity) = pool.getRedemptionBalance(_args.otoken);
            uint256 newQuantity = oldQuantity.sub(_args.amount);

            if(otoken.isPut()) {
                uint256 oldCollateral = oldQuantity.mul(otoken.strikePrice()).mul(10**collateralDecimals).div(1e16);
                uint256 newCollateral = newQuantity.mul(otoken.strikePrice()).mul(10**collateralDecimals).div(1e16);
                pool.updateRedemptionBalance(
                    _args.otoken,
                    -int256(_args.amount),
                    false,
                    -int256(oldCollateral.sub(newCollateral))
                );
            } else {
                uint256 oldCollateral = oldQuantity.mul(10**collateralDecimals).div(1e8);
                uint256 newCollateral = newQuantity.mul(10**collateralDecimals).div(1e8);
                pool.updateRedemptionBalance(
                    _args.otoken,
                    -int256(_args.amount),
                    false,
                    -int256(oldCollateral.sub(newCollateral))
                );
            }

        } else {
            // this is not a physically settled vault
            require(!otoken.isPhysicallySettled(), "C40");
        }

        // remove otoken from vault
        controller.updateVault(1, _args.owner, _args.vaultId, _args.otoken, _args.amount);
        // burn otoken
        otoken.burnOtoken(_args.from, _args.amount);

        emit ShortOtokenBurned(_args.otoken, _args.owner, _args.from, _args.vaultId, _args.amount);
    }

    function handleRedeem(Actions.RedeemArgs memory _args, address sender) external onlyController {
        OtokenInterface otoken = OtokenInterface(_args.otoken);

        // check that otoken to redeem is whitelisted
        require(whitelist.isWhitelistedOtoken(_args.otoken), "C27");

        (address collateral, address underlying, address strike, uint256 expiry) = _getOtokenDetails(address(otoken));

        // only allow redeeming expired otoken
        require(now >= expiry, "C28");

        require(controller.canSettleAssets(underlying, strike, collateral, expiry), "C29");

        // prevent truncation in payout/payment calculations for tokens with fewer decimals than oTokens (8)
        uint256 collateralDecimals = uint256(ERC20Interface(collateral).decimals());
        if (collateralDecimals < 8) {
            require(_args.amount % (10**(8 - collateralDecimals)) == 0, "C46");
        }

        if (otoken.isPhysicallySettled()) {
            // must be within the redeem time period
            require(now < expiry + redeemTimePeriod, "C39");
            
            uint256 strikePrice = otoken.strikePrice();
            (uint256 underlyingExpiryPrice,) = oracle.getExpiryPrice(underlying, expiry);
            (uint256 collateralExpiryPrice,) = oracle.getExpiryPrice(collateral, expiry);

            // get the amount of capital needed to exercise.
            // for calls this is strikeAsset, for puts it is underlyingAsset
            uint256 strikePayment = calculator.getStrikePaymentAmount(_args.otoken, _args.amount); 

            if(otoken.isPut()) {
                // check put is ITM
                require(underlyingExpiryPrice < strikePrice, "C45");
                // take the underlying asset payment
                _removeExcessCollateralFromRedemptionBalances(otoken, strikePrice, underlyingExpiryPrice, collateralExpiryPrice);
                pool.transferToPool(otoken.underlyingAsset(), _args.receiver, strikePayment);
            } else {
                // check call is ITM
                require(underlyingExpiryPrice > strikePrice, "C45");

                _removeExcessCollateralFromRedemptionBalances(otoken, strikePrice, underlyingExpiryPrice, collateralExpiryPrice);
                // take the strike asset payment (zero-strike calls have no payment so skip to avoid MarginPool revert)
                if (strikePayment > 0) {
                    pool.transferToPool(otoken.strikeAsset(), _args.receiver, strikePayment);
                }
            }
            
            pool.updateRedemptionBalance(_args.otoken, 0, true, int256(strikePayment));
        }
        uint256 payout = getPayout(_args.otoken, _args.amount);
        otoken.burnOtoken(sender, _args.amount);

        pool.transferToUser(collateral, _args.receiver, payout);
        
        if (otoken.isPhysicallySettled()) {
            pool.updateRedemptionBalance(_args.otoken, 0, false, -int256(payout));
        }

        emit Redeem(_args.otoken, sender, _args.receiver, collateral, _args.amount, payout);
    }

    function handleSettle(Actions.SettleVaultArgs memory _args) external onlyController {
        (MarginVault.Vault memory vault, uint256 typeVault, ) = controller.getVaultWithDetails(
            _args.owner,
            _args.vaultId
        );

        // check if there is short or long otoken in vault
        // do not allow settling vault that have no short or long otoken
        // if there is a long otoken, burn it
        // store otoken address outside of this scope
        
        bool hasShort = _isNotEmpty(vault.shortOtokens);
        bool hasLong = _isNotEmpty(vault.longOtokens);

        require(hasShort || hasLong, "C30");

        OtokenInterface otoken = hasShort ? OtokenInterface(vault.shortOtokens[0]) : OtokenInterface(vault.longOtokens[0]);

        if (hasLong) {
            OtokenInterface longOtoken = OtokenInterface(vault.longOtokens[0]);

            longOtoken.burnOtoken(address(pool), vault.longAmounts[0]);
            pool.decrementBalanceAfterBurn(vault.longOtokens[0], vault.longAmounts[0]);
        }

        SettleMem memory settleMem;
        ( settleMem.collateral, settleMem.underlying, settleMem.strike, settleMem.expiry) = _getOtokenDetails(address(otoken));

        // do not allow settling vault with un-expired otoken
        if (typeVault == 2) {
            require(now >= settleMem.expiry + redeemTimePeriod, "C41");
        } else {
            require(now >= settleMem.expiry, "C31");
        }
        require(controller.canSettleAssets(settleMem.underlying, settleMem.strike, settleMem.collateral, settleMem.expiry), "C29");

        // getExcessCollateral calculates collateral obligations differently post-expiry depending on settlement type (see _getCashValue):
        // - Cash-settled: the obligation is the cash value of the option (max(underlyingPrice - strike, 0) for calls). All collateral
        //   beyond this obligation is excess and can be returned to the writer.
        // - Physically-settled: the obligation is the amount of collateral that would be redeemed by the option buyer in exchange for
        //   the strike payment (underlyingPrice for ITM calls, strikePrice for ITM puts). This is larger than the cash value because
        //   it represents the full delivery amount, not just the profit.
        // For OTM options (both types), the obligation is zero — the full collateral amount is returned as excess.
        (settleMem.collateralPayout, settleMem.isValidVault) = calculator.getExcessCollateral(vault, typeVault);

        // require that vault is valid (has excess collateral) before settling.
        // the calculator's peg-equivalent depeg cap ensures that under LST depeg, the per-oToken obligation
        // is capped at the pegged rate, so properly collateralized vaults always report as valid.
        require(settleMem.isValidVault, "C32");

        // For physically settled vaults, the writer is also entitled to their pro-rata share of the
        // redemptionBalances pool for this oToken series. This pool tracks both unredeemed collateral
        // (from options that expired unexercised) and strike payments received from buyers who exercised.
        // The following block calculates the writer's share of these balances, added on top of any
        // excess collateral already computed above.
        if (typeVault == 2) {
            uint256 strikePrice = otoken.strikePrice();
            (settleMem.underlyingExpiryPrice,) = oracle.getExpiryPrice(settleMem.underlying, settleMem.expiry);
            (settleMem.collateralExpiryPrice,) = oracle.getExpiryPrice(settleMem.collateral, settleMem.expiry);

            _removeExcessCollateralFromRedemptionBalances(
                otoken, 
                strikePrice,
                settleMem.underlyingExpiryPrice,
                settleMem.collateralExpiryPrice
            );
            
            (settleMem.collateralRedemptionBalance, settleMem.receivingAssetBalance, settleMem.otokenQuantity) = pool.getRedemptionBalance(address(otoken));

            settleMem.shortAmount = vault.shortAmounts[0];

            // CALLS:
            // strike payment is number of options in vault * strike price, converted to strike decimals
            // strikePrice denominatd in e8, shortAmount denominated in e8
            // strikePrice * shortAmount * e2 = e18
            // divide by (18 - strike decimals) to convert to strike decimals
            // assumes strike asset decimals <= 18
            // PUTS:
            // strike payment is number of options in vault, converted to underying decimals
            // shortAmount * e10 = e18
            // divide by (18 - underlying decimals) to convert to underlying decimals
            settleMem.strikePayout = otoken.isPut() ? 
                settleMem.shortAmount.mul(1e10).div(10**(18 - uint256(ERC20Interface(settleMem.underlying).decimals()))) :
                (strikePrice.mul(settleMem.shortAmount).mul(1e2)).div(10**(18 - uint256(ERC20Interface(settleMem.strike).decimals()))
            );

            // From here there are three scenarios that can happen:
            // 0. (IF OTM) There is no strike asset in the pool. Meaning the user should receive their payment in collateral + any collateral they have from overcollateralisation
            // 1. (IF ITM and no strike asset in the pool) Meaning the user should receive their payment in collateral + any collateral they have from overcollateralisation 
            // 2. (IF ITM not enough strike asset in the pool to pay the strike payment) Meaning the user should receive any remaining strike asset, their remaining payment in collateral + any collateral they have from overcollateralisation
            // 3. (IF ITM and there is enough strike asset in the pool to pay the strike payment) Meaning the user should receive their payment in strike + any collateral they have from overcollateralization
            if (settleMem.receivingAssetBalance == 0) {
                // this is scenario 0 or 1
                settleMem.strikePayout = 0;
                // each remaining otoken to settle gets an equal share of collateralRedemptionBalance, (for OTM options this is zero)
                uint256 vaultShareOfCollateralRedemptionBalance = settleMem.collateralRedemptionBalance.mul(settleMem.shortAmount).div(settleMem.otokenQuantity);
                settleMem.collateralPayout += vaultShareOfCollateralRedemptionBalance;

                pool.updateRedemptionBalance(address(otoken), -int256(settleMem.shortAmount), false, -int256(vaultShareOfCollateralRedemptionBalance));
            } else if (settleMem.strikePayout > settleMem.receivingAssetBalance) {
                // this is scenario 2
                settleMem.strikePayout = settleMem.receivingAssetBalance;
                // find how many options worth of strike payment is in the pool
                // CALLS: get receivingAssetBalance and convert to e18 notation then divide by strikePrice * 1e2 (e10) to get e8 value.
                // PUTS: get receivingAssetBalance and convert to e8 notation
                // assumes strike asset decimals <= 18
                settleMem.strikeCount = otoken.isPut() ? 
                    settleMem.receivingAssetBalance.mul(10**(18 - uint256(ERC20Interface(settleMem.underlying).decimals()))).div(1e10) :
                    settleMem.receivingAssetBalance.mul(10**(18 - uint256(ERC20Interface(settleMem.strike).decimals()))).div(strikePrice.mul(1e2));
                // we can get the amount the user is owed in collateral asset by subtracting the strikeCount from the shortAmounts
                uint256 contractsLeft = settleMem.shortAmount.sub(settleMem.strikeCount);
                // divide collateralRedemptionBalance by total number of options to split between (otokenQuantity - strikeCount) and multiply by contractsLeft
                uint256 vaultShareOfCollateralRedemptionBalance = settleMem.collateralRedemptionBalance.mul(contractsLeft).div(settleMem.otokenQuantity.sub(settleMem.strikeCount));
                // the collateral payout is whatever it already was + the share of the collateralRedemptionBalance
                settleMem.collateralPayout += vaultShareOfCollateralRedemptionBalance;
                pool.updateRedemptionBalance(address(otoken), -int256(settleMem.strikeCount), true, -int256(settleMem.strikePayout));
                pool.updateRedemptionBalance(address(otoken), -int256(contractsLeft), false, -int256(vaultShareOfCollateralRedemptionBalance));
            } else {
                // this is scenario 3
                pool.updateRedemptionBalance(address(otoken), -int256(settleMem.shortAmount), true, -int256(settleMem.strikePayout));
            }
        }

        controller.updateVault(6, _args.owner, _args.vaultId, address(0), 0);

        if (settleMem.collateralPayout > 0) {
            pool.transferToUser(settleMem.collateral, _args.to, settleMem.collateralPayout);
        }
        if (settleMem.strikePayout > 0) {
            pool.transferToUser(otoken.isPut() ? otoken.underlyingAsset() : otoken.strikeAsset(), _args.to, settleMem.strikePayout);
        }
        uint256 vaultId = _args.vaultId;
        address payoutRecipient = _args.to;

        emit VaultSettled(_args.owner, address(otoken), payoutRecipient, settleMem.collateralPayout, settleMem.strikePayout, vaultId, typeVault);
    }

    function _removeExcessCollateralFromRedemptionBalances(
         OtokenInterface otoken,
         uint256 strikePrice,
         uint256 underlyingExpiryPrice,
         uint256 collateralExpiryPrice
    ) internal {
        // At mint time, collateralRedemptionBalance is set to 1 unit of collateral per oToken (for calls). But after expiry,
        // not all of that collateral is owed to option buyers — only the portion worth 1 unit of underlying (the delivery obligation).
        // If the collateral is yield-bearing (e.g. kHYPE > HYPE), there is excess collateral per option beyond the delivery amount.
        // This function removes that excess once, so the remaining collateralRedemptionBalance can be split pro-rata among
        // writers whose vaults were not fully paid out in strike asset (scenarios 0/1/2 in handleSettle).
        
        (uint256 collateralRedemptionBalance, , uint256 otokenQuantity) = pool.getRedemptionBalance(address(otoken));
        uint256 totalExcessInPool;

        if(
            otoken.isPut() && underlyingExpiryPrice >= strikePrice ||
            !otoken.isPut() && underlyingExpiryPrice <= strikePrice
        ){
            // options expired OTM
            // no receiving asset consideration, vaults receive all their collateral back
            // set collateralRedemptionBalance to 0 since calculator.getExcessCollateral handles this case (because short options are worthless)
            // return if already set to 0
            if (collateralRedemptionBalance == 0) return;

            totalExcessInPool = collateralRedemptionBalance;
        } else if(otoken.isPut()) {
            // dont need to handle ITM put cases because collateralAsset == strikeAsset, so fully collateralised with no excess.
            return;
        } else if(!otoken.isPut()) {
            // ITM calls
            // first, check to see if this has already been done. We only want to do it once, the first time a settle OR redeem is done.
            // if is has NOT been done, collateralRedemptionBalance == otokenQuantity (accounting for decimals).
            // so we can return if this is not the case because it has already been done

            uint256 otokenQuantityCollateralDecimals = otokenQuantity.mul(10**uint256(ERC20Interface(otoken.collateralAsset()).decimals())).div(1e8);

            if (collateralRedemptionBalance != otokenQuantityCollateralDecimals) return;
            
            // calls expired ITM
            // excess collateral for each option in pool is 1 - underlyingPrice/collateralPrice
            // if underlyingPrice >= collateralPrice, there is no excess: set to 0
            totalExcessInPool = underlyingExpiryPrice < collateralExpiryPrice ? 
                otokenQuantityCollateralDecimals.sub(otokenQuantityCollateralDecimals.mul(underlyingExpiryPrice).div(collateralExpiryPrice)) :
                0;    
        }

        // reduce value in redemptionBalancePool
        pool.updateRedemptionBalance(
            address(otoken), 
            0, 
            false, 
            -int(totalExcessInPool)
        );
    }

    /**
     * @dev get otoken detail, from both otoken versions
     */
    function _getOtokenDetails(address _otoken)
        internal
        view
        returns (
            address,
            address,
            address,
            uint256
        )
    {
        OtokenInterface otoken = OtokenInterface(_otoken);
        (address collateral, address underlying, address strike, , uint256 expiry, , ) = otoken.getOtokenDetails();
        return (collateral, underlying, strike, expiry);
    }

    function _isNotEmpty(address[] memory _array) internal pure returns (bool) {
        return (_array.length > 0) && (_array[0] != address(0));
    }

    /**
     * @dev updates the internal configuration of the controller
     */
    function _refreshConfigInternal() internal {
        whitelist = WhitelistInterface(addressbook.getWhitelist());
        controller = ControllerInterface(addressbook.getController());
        oracle = OracleInterface(addressbook.getOracle());
        calculator = MarginCalculatorInterface(addressbook.getMarginCalculator());
        pool = MarginPoolInterface(addressbook.getMarginPool());
    }
}




// ===== FILE: contracts/packages/oz/upgradeability/OwnableUpgradeSafe.sol =====
// SPDX-License-Identifier: MIT
// openzeppelin-contracts-upgradeable v3.0.0

pragma solidity ^0.6.0;

import "./GSN/ContextUpgradeable.sol";
import "./Initializable.sol";

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
contract OwnableUpgradeSafe is Initializable, ContextUpgradeable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */

    function __Ownable_init(address _sender) internal initializer {
        __Context_init_unchained();
        __Ownable_init_unchained(_sender);
    }

    function __Ownable_init_unchained(address _sender) internal initializer {
        _owner = _sender;
        emit OwnershipTransferred(address(0), _sender);
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions anymore. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }

    uint256[49] private __gap;
}


// ===== FILE: contracts/packages/oz/upgradeability/ReentrancyGuardUpgradeSafe.sol =====
// SPDX-License-Identifier: MIT
// openzeppelin-contracts-upgradeable v3.0.0

pragma solidity ^0.6.0;

import "./Initializable.sol";

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
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
contract ReentrancyGuardUpgradeSafe is Initializable {
    bool private _notEntered;

    function __ReentrancyGuard_init() internal initializer {
        __ReentrancyGuard_init_unchained();
    }

    function __ReentrancyGuard_init_unchained() internal initializer {
        // Storing an initial non-zero value makes deployment a bit more
        // expensive, but in exchange the refund on every call to nonReentrant
        // will be lower in amount. Since refunds are capped to a percetange of
        // the total transaction's gas, it is best to keep them low in cases
        // like this one, to increase the likelihood of the full refund coming
        // into effect.
        _notEntered = true;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and make it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        // On the first call to nonReentrant, _notEntered will be true
        require(_notEntered, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _notEntered = false;

        _;

        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _notEntered = true;
    }

    uint256[49] private __gap;
}


// ===== FILE: contracts/packages/oz/upgradeability/Initializable.sol =====
// SPDX-License-Identifier: MIT
// openzeppelin-contracts-upgradeable v3.0.0

/* solhint-disable */
pragma solidity >=0.4.24 <0.7.0;

/**
 * @title Initializable
 *
 * @dev Helper contract to support initializer functions. To use it, replace
 * the constructor with a function that has the `initializer` modifier.
 * WARNING: Unlike constructors, initializer functions must be manually
 * invoked. This applies both to deploying an Initializable contract, as well
 * as extending an Initializable contract via inheritance.
 * WARNING: When used with inheritance, manual care must be taken to not invoke
 * a parent initializer twice, or ensure that all initializers are idempotent,
 * because this is not dealt with automatically as with constructors.
 */
contract Initializable {
    /**
     * @dev Indicates that the contract has been initialized.
     */
    bool private initialized;

    /**
     * @dev Indicates that the contract is in the process of being initialized.
     */
    bool private initializing;

    /**
     * @dev Modifier to use in the initializer function of a contract.
     */
    modifier initializer() {
        require(initializing || isConstructor() || !initialized, "Contract instance has already been initialized");

        bool isTopLevelCall = !initializing;
        if (isTopLevelCall) {
            initializing = true;
            initialized = true;
        }

        _;

        if (isTopLevelCall) {
            initializing = false;
        }
    }

    /// @dev Returns true if and only if the function is running in the constructor
    function isConstructor() private view returns (bool) {
        // extcodesize checks the size of the code stored in an address, and
        // address returns the current address. Since the code is still not
        // deployed when running a constructor, any checks on its code size will
        // yield zero, making it an effective way to detect if a contract is
        // under construction or not.
        address self = address(this);
        uint256 cs;
        assembly {
            cs := extcodesize(self)
        }
        return cs == 0;
    }

    // Reserved storage space to allow for layout changes in the future.
    uint256[50] private ______gap;
}


// ===== FILE: contracts/packages/oz/SafeMath.sol =====
// SPDX-License-Identifier: MIT
// openzeppelin-contracts v3.1.0

/* solhint-disable */
pragma solidity ^0.6.0;

/**
 * @dev Wrappers over Solidity's arithmetic operations with added overflow
 * checks.
 *
 * Arithmetic operations in Solidity wrap on overflow. This can easily result
 * in bugs, because programmers usually assume that an overflow raises an
 * error, which is the standard behavior in high level programming languages.
 * `SafeMath` restores this intuition by reverting the transaction when an
 * operation overflows.
 *
 * Using this library instead of the unchecked operations eliminates an entire
 * class of bugs, so it's recommended to use it always.
 */
library SafeMath {
    /**
     * @dev Returns the addition of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `+` operator.
     *
     * Requirements:
     * - Addition cannot overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");

        return c;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting on
     * overflow (when the result is negative).
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting with custom message on
     * overflow (when the result is negative).
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     * - Subtraction cannot overflow.
     */
    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;

        return c;
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `*` operator.
     *
     * Requirements:
     * - Multiplication cannot overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
        // benefit is lost if 'b' is also tested.
        // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");

        return c;
    }

    /**
     * @dev Returns the integer division of two unsigned integers. Reverts on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    /**
     * @dev Returns the integer division of two unsigned integers. Reverts with custom message on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     * - The divisor cannot be zero.
     */
    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        // Solidity only automatically asserts when dividing by 0
        require(b > 0, errorMessage);
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold

        return c;
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * Reverts when dividing by zero.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return mod(a, b, "SafeMath: modulo by zero");
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * Reverts with custom message when dividing by zero.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     * - The divisor cannot be zero.
     */
    function mod(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b != 0, errorMessage);
        return a % b;
    }
}


// ===== FILE: contracts/libs/MarginVault.sol =====
/**
 * SPDX-License-Identifier: UNLICENSED
 */
pragma solidity =0.6.10;

pragma experimental ABIEncoderV2;

import {SafeMath} from "../packages/oz/SafeMath.sol";

/**
 * MarginVault Error Codes
 * V1: invalid short otoken amount
 * V2: invalid short otoken index
 * V3: short otoken address mismatch
 * V4: invalid long otoken amount
 * V5: invalid long otoken index
 * V6: long otoken address mismatch
 * V7: invalid collateral amount
 * V8: invalid collateral token index
 * V9: collateral token address mismatch
 */

/**
 * @title MarginVault
 * @author Opyn Team
 * @notice A library that provides the Controller with a Vault struct and the functions that manipulate vaults.
 * Vaults describe discrete position combinations of long options, short options, and collateral assets that a user can have.
 */
library MarginVault {
    using SafeMath for uint256;

    // vault is a struct of 6 arrays that describe a position a user has, a user can have multiple vaults.
    struct Vault {
        // addresses of oTokens a user has shorted (i.e. written) against this vault
        address[] shortOtokens;
        // addresses of oTokens a user has bought and deposited in this vault
        // user can be long oTokens without opening a vault (e.g. by buying on a DEX)
        // generally, long oTokens will be 'deposited' in vaults to act as collateral in order to write oTokens against (i.e. in spreads)
        address[] longOtokens;
        // addresses of other ERC-20s a user has deposited as collateral in this vault
        address[] collateralAssets;
        // quantity of oTokens minted/written for each oToken address in shortOtokens
        uint256[] shortAmounts;
        // quantity of oTokens owned and held in the vault for each oToken address in longOtokens
        uint256[] longAmounts;
        // quantity of ERC-20 deposited as collateral in the vault for each ERC-20 address in collateralAssets
        uint256[] collateralAmounts;
    }

    /**
     * @dev increase the short oToken balance in a vault when a new oToken is minted
     * @param _vault vault to add or increase the short position in
     * @param _shortOtoken address of the _shortOtoken being minted from the user's vault
     * @param _amount number of _shortOtoken being minted from the user's vault
     * @param _index index of _shortOtoken in the user's vault.shortOtokens array
     */
    function addShort(
        Vault storage _vault,
        address _shortOtoken,
        uint256 _amount,
        uint256 _index
    ) external {
        require(_amount > 0, "V1");

        // valid indexes in any array are between 0 and array.length - 1.
        // if adding an amount to an preexisting short oToken, check that _index is in the range of 0->length-1
        if ((_index == _vault.shortOtokens.length) && (_index == _vault.shortAmounts.length)) {
            _vault.shortOtokens.push(_shortOtoken);
            _vault.shortAmounts.push(_amount);
        } else {
            require((_index < _vault.shortOtokens.length) && (_index < _vault.shortAmounts.length), "V2");
            address existingShort = _vault.shortOtokens[_index];
            require((existingShort == _shortOtoken) || (existingShort == address(0)), "V3");

            _vault.shortAmounts[_index] = _vault.shortAmounts[_index].add(_amount);
            _vault.shortOtokens[_index] = _shortOtoken;
        }
    }

    /**
     * @dev decrease the short oToken balance in a vault when an oToken is burned
     * @param _vault vault to decrease short position in
     * @param _shortOtoken address of the _shortOtoken being reduced in the user's vault
     * @param _amount number of _shortOtoken being reduced in the user's vault
     * @param _index index of _shortOtoken in the user's vault.shortOtokens array
     */
    function removeShort(
        Vault storage _vault,
        address _shortOtoken,
        uint256 _amount,
        uint256 _index
    ) external {
        // check that the removed short oToken exists in the vault at the specified index
        require(_index < _vault.shortOtokens.length, "V2");
        require(_vault.shortOtokens[_index] == _shortOtoken, "V3");

        uint256 newShortAmount = _vault.shortAmounts[_index].sub(_amount);

        if (newShortAmount == 0) {
            delete _vault.shortOtokens[_index];
        }
        _vault.shortAmounts[_index] = newShortAmount;
    }

    /**
     * @dev increase the long oToken balance in a vault when an oToken is deposited
     * @param _vault vault to add a long position to
     * @param _longOtoken address of the _longOtoken being added to the user's vault
     * @param _amount number of _longOtoken the protocol is adding to the user's vault
     * @param _index index of _longOtoken in the user's vault.longOtokens array
     */
    function addLong(
        Vault storage _vault,
        address _longOtoken,
        uint256 _amount,
        uint256 _index
    ) external {
        require(_amount > 0, "V4");

        // valid indexes in any array are between 0 and array.length - 1.
        // if adding an amount to an preexisting short oToken, check that _index is in the range of 0->length-1
        if ((_index == _vault.longOtokens.length) && (_index == _vault.longAmounts.length)) {
            _vault.longOtokens.push(_longOtoken);
            _vault.longAmounts.push(_amount);
        } else {
            require((_index < _vault.longOtokens.length) && (_index < _vault.longAmounts.length), "V5");
            address existingLong = _vault.longOtokens[_index];
            require((existingLong == _longOtoken) || (existingLong == address(0)), "V6");

            _vault.longAmounts[_index] = _vault.longAmounts[_index].add(_amount);
            _vault.longOtokens[_index] = _longOtoken;
        }
    }

    /**
     * @dev decrease the long oToken balance in a vault when an oToken is withdrawn
     * @param _vault vault to remove a long position from
     * @param _longOtoken address of the _longOtoken being removed from the user's vault
     * @param _amount number of _longOtoken the protocol is removing from the user's vault
     * @param _index index of _longOtoken in the user's vault.longOtokens array
     */
    function removeLong(
        Vault storage _vault,
        address _longOtoken,
        uint256 _amount,
        uint256 _index
    ) external {
        // check that the removed long oToken exists in the vault at the specified index
        require(_index < _vault.longOtokens.length, "V5");
        require(_vault.longOtokens[_index] == _longOtoken, "V6");

        uint256 newLongAmount = _vault.longAmounts[_index].sub(_amount);

        if (newLongAmount == 0) {
            delete _vault.longOtokens[_index];
        }
        _vault.longAmounts[_index] = newLongAmount;
    }

    /**
     * @dev increase the collateral balance in a vault
     * @param _vault vault to add collateral to
     * @param _collateralAsset address of the _collateralAsset being added to the user's vault
     * @param _amount number of _collateralAsset being added to the user's vault
     * @param _index index of _collateralAsset in the user's vault.collateralAssets array
     */
    function addCollateral(
        Vault storage _vault,
        address _collateralAsset,
        uint256 _amount,
        uint256 _index
    ) external {
        require(_amount > 0, "V7");

        // valid indexes in any array are between 0 and array.length - 1.
        // if adding an amount to an preexisting short oToken, check that _index is in the range of 0->length-1
        if ((_index == _vault.collateralAssets.length) && (_index == _vault.collateralAmounts.length)) {
            _vault.collateralAssets.push(_collateralAsset);
            _vault.collateralAmounts.push(_amount);
        } else {
            require((_index < _vault.collateralAssets.length) && (_index < _vault.collateralAmounts.length), "V8");
            address existingCollateral = _vault.collateralAssets[_index];
            require((existingCollateral == _collateralAsset) || (existingCollateral == address(0)), "V9");

            _vault.collateralAmounts[_index] = _vault.collateralAmounts[_index].add(_amount);
            _vault.collateralAssets[_index] = _collateralAsset;
        }
    }

    /**
     * @dev decrease the collateral balance in a vault
     * @param _vault vault to remove collateral from
     * @param _collateralAsset address of the _collateralAsset being removed from the user's vault
     * @param _amount number of _collateralAsset being removed from the user's vault
     * @param _index index of _collateralAsset in the user's vault.collateralAssets array
     */
    function removeCollateral(
        Vault storage _vault,
        address _collateralAsset,
        uint256 _amount,
        uint256 _index
    ) external {
        // check that the removed collateral exists in the vault at the specified index
        require(_index < _vault.collateralAssets.length, "V8");
        require(_vault.collateralAssets[_index] == _collateralAsset, "V9");

        uint256 newCollateralAmount = _vault.collateralAmounts[_index].sub(_amount);

        if (newCollateralAmount == 0) {
            delete _vault.collateralAssets[_index];
        }
        _vault.collateralAmounts[_index] = newCollateralAmount;
    }
}


// ===== FILE: contracts/libs/Actions.sol =====
/**
 * SPDX-License-Identifier: UNLICENSED
 */
pragma solidity 0.6.10;

import {MarginVault} from "./MarginVault.sol";

/**
 * @title Actions
 * @author Opyn Team
 * @notice A library that provides a ActionArgs struct, sub types of Action structs, and functions to parse ActionArgs into specific Actions.
 * errorCode
 * A1 can only parse arguments for open vault actions
 * A2 cannot open vault for an invalid account
 * A3 cannot open vault with an invalid type
 * A4 can only parse arguments for mint actions
 * A5 cannot mint from an invalid account
 * A6 can only parse arguments for burn actions
 * A7 cannot burn from an invalid account
 * A8 can only parse arguments for deposit actions
 * A9 cannot deposit to an invalid account
 * A10 can only parse arguments for withdraw actions
 * A11 cannot withdraw from an invalid account
 * A12 cannot withdraw to an invalid account
 * A13 can only parse arguments for redeem actions
 * A14 cannot redeem to an invalid account
 * A15 can only parse arguments for settle vault actions
 * A16 cannot settle vault for an invalid account
 * A17 cannot withdraw payout to an invalid account
 * A18 can only parse arguments for liquidate action
 * A19 cannot liquidate vault for an invalid account owner
 * A20 cannot send collateral to an invalid account
 * A21 cannot parse liquidate action with no round id
 * A22 can only parse arguments for call actions
 * A23 target address cannot be address(0)
 */
library Actions {
    // possible actions that can be performed
    enum ActionType {
        OpenVault,
        MintShortOption,
        BurnShortOption,
        DepositLongOption,
        WithdrawLongOption,
        DepositCollateral,
        WithdrawCollateral,
        SettleVault,
        Redeem,
        Call, // DEPRECATED
        Liquidate // DEPRECATED
    }

    struct ActionArgs {
        // type of action that is being performed on the system
        ActionType actionType;
        // address of the account owner
        address owner;
        // address which we move assets from or to (depending on the action type)
        address secondAddress;
        // asset that is to be transfered
        address asset;
        // index of the vault that is to be modified (if any)
        uint256 vaultId;
        // amount of asset that is to be transfered
        uint256 amount;
        // each vault can hold multiple short / long / collateral assets but we are restricting the scope to only 1 of each in this version
        // in future versions this would be the index of the short / long / collateral asset that needs to be modified
        uint256 index;
        // any other data that needs to be passed in for arbitrary function calls
        bytes data;
    }

    struct MintArgs {
        // address of the account owner
        address owner;
        // index of the vault from which the asset will be minted
        uint256 vaultId;
        // address to which we transfer the minted oTokens
        address to;
        // oToken that is to be minted
        address otoken;
        // each vault can hold multiple short / long / collateral assets but we are restricting the scope to only 1 of each in this version
        // in future versions this would be the index of the short / long / collateral asset that needs to be modified
        uint256 index;
        // amount of oTokens that is to be minted
        uint256 amount;
    }

    struct BurnArgs {
        // address of the account owner
        address owner;
        // index of the vault from which the oToken will be burned
        uint256 vaultId;
        // address from which we transfer the oTokens
        address from;
        // oToken that is to be burned
        address otoken;
        // each vault can hold multiple short / long / collateral assets but we are restricting the scope to only 1 of each in this version
        // in future versions this would be the index of the short / long / collateral asset that needs to be modified
        uint256 index;
        // amount of oTokens that is to be burned
        uint256 amount;
    }

    struct OpenVaultArgs {
        // address of the account owner
        address owner;
        // vault id to create
        uint256 vaultId;
        // vault type, 0 for spread/max loss, 2 for physially settled
        uint256 vaultType;
    }

    struct DepositArgs {
        // address of the account owner
        address owner;
        // index of the vault to which the asset will be added
        uint256 vaultId;
        // address from which we transfer the asset
        address from;
        // asset that is to be deposited
        address asset;
        // each vault can hold multiple short / long / collateral assets but we are restricting the scope to only 1 of each in this version
        // in future versions this would be the index of the short / long / collateral asset that needs to be modified
        uint256 index;
        // amount of asset that is to be deposited
        uint256 amount;
    }

    struct RedeemArgs {
        // address to which we pay out the oToken proceeds
        address receiver;
        // oToken that is to be redeemed
        address otoken;
        // amount of oTokens that is to be redeemed
        uint256 amount;
    }

    struct WithdrawArgs {
        // address of the account owner
        address owner;
        // index of the vault from which the asset will be withdrawn
        uint256 vaultId;
        // address to which we transfer the asset
        address to;
        // asset that is to be withdrawn
        address asset;
        // each vault can hold multiple short / long / collateral assets but we are restricting the scope to only 1 of each in this version
        // in future versions this would be the index of the short / long / collateral asset that needs to be modified
        uint256 index;
        // amount of asset that is to be withdrawn
        uint256 amount;
    }

    struct SettleVaultArgs {
        // address of the account owner
        address owner;
        // index of the vault to which is to be settled
        uint256 vaultId;
        // address to which we transfer the remaining collateral
        address to;
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for an open vault action
     * @param _args general action arguments structure
     * @return arguments for a open vault action
     */
    function _parseOpenVaultArgs(ActionArgs memory _args) internal pure returns (OpenVaultArgs memory) {
        require(_args.actionType == ActionType.OpenVault, "A1");
        require(_args.owner != address(0), "A2");

        // if not _args.data included, vault type will be 0 by default
        uint256 vaultType;

        if (_args.data.length == 32) {
            // decode vault type from _args.data
            vaultType = abi.decode(_args.data, (uint256));
        }

        // we only have fully collateralised/spread vaults (type 0) and  physically settled vaults (type 2) in this version
        require(vaultType == 0 || vaultType == 2, "A3");

        return OpenVaultArgs({owner: _args.owner, vaultId: _args.vaultId, vaultType: vaultType});
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for a mint action
     * @param _args general action arguments structure
     * @return arguments for a mint action
     */
    function _parseMintArgs(ActionArgs memory _args) internal pure returns (MintArgs memory) {
        require(_args.actionType == ActionType.MintShortOption, "A4");
        require(_args.owner != address(0), "A5");

        return
            MintArgs({
                owner: _args.owner,
                vaultId: _args.vaultId,
                to: _args.secondAddress,
                otoken: _args.asset,
                index: _args.index,
                amount: _args.amount
            });
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for a burn action
     * @param _args general action arguments structure
     * @return arguments for a burn action
     */
    function _parseBurnArgs(ActionArgs memory _args) internal pure returns (BurnArgs memory) {
        require(_args.actionType == ActionType.BurnShortOption, "A6");
        require(_args.owner != address(0), "A7");

        return
            BurnArgs({
                owner: _args.owner,
                vaultId: _args.vaultId,
                from: _args.secondAddress,
                otoken: _args.asset,
                index: _args.index,
                amount: _args.amount
            });
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for a deposit action
     * @param _args general action arguments structure
     * @return arguments for a deposit action
     */
    function _parseDepositArgs(ActionArgs memory _args) internal pure returns (DepositArgs memory) {
        require(
            (_args.actionType == ActionType.DepositLongOption) || (_args.actionType == ActionType.DepositCollateral),
            "A8"
        );
        require(_args.owner != address(0), "A9");

        return
            DepositArgs({
                owner: _args.owner,
                vaultId: _args.vaultId,
                from: _args.secondAddress,
                asset: _args.asset,
                index: _args.index,
                amount: _args.amount
            });
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for a withdraw action
     * @param _args general action arguments structure
     * @return arguments for a withdraw action
     */
    function _parseWithdrawArgs(ActionArgs memory _args) internal pure returns (WithdrawArgs memory) {
        require(
            (_args.actionType == ActionType.WithdrawLongOption) || (_args.actionType == ActionType.WithdrawCollateral),
            "A10"
        );
        require(_args.owner != address(0), "A11");
        require(_args.secondAddress != address(0), "A12");

        return
            WithdrawArgs({
                owner: _args.owner,
                vaultId: _args.vaultId,
                to: _args.secondAddress,
                asset: _args.asset,
                index: _args.index,
                amount: _args.amount
            });
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for an redeem action
     * @param _args general action arguments structure
     * @return arguments for a redeem action
     */
    function _parseRedeemArgs(ActionArgs memory _args) internal pure returns (RedeemArgs memory) {
        require(_args.actionType == ActionType.Redeem, "A13");
        require(_args.secondAddress != address(0), "A14");

        return RedeemArgs({receiver: _args.secondAddress, otoken: _args.asset, amount: _args.amount});
    }

    /**
     * @notice parses the passed in action arguments to get the arguments for a settle vault action
     * @param _args general action arguments structure
     * @return arguments for a settle vault action
     */
    function _parseSettleVaultArgs(ActionArgs memory _args) internal pure returns (SettleVaultArgs memory) {
        require(_args.actionType == ActionType.SettleVault, "A15");
        require(_args.owner != address(0), "A16");
        require(_args.secondAddress != address(0), "A17");

        return SettleVaultArgs({owner: _args.owner, vaultId: _args.vaultId, to: _args.secondAddress});
    }
}


// ===== FILE: contracts/interfaces/ERC20Interface.sol =====
/**
 * SPDX-License-Identifier: UNLICENSED
 */
pragma solidity 0.6.10;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface ERC20Interface {
    /**
     * @dev Returns the amount of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `recipient`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
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
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `sender` to `recipient` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    function decimals() external view returns (uint8);

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
}


// ===== FILE: contracts/interfaces/AddressBookInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

interface AddressBookInterface {
    /* Getters */

    function getOtokenImpl() external view returns (address);

    function getOtokenFactory() external view returns (address);

    function getWhitelist() external view returns (address);

    function getController() external view returns (address);

    function getOracle() external view returns (address);

    function getMarginPool() external view returns (address);

    function getMarginCalculator() external view returns (address);

    function getControllerLogic() external view returns (address);

    function getAddress(bytes32 _id) external view returns (address);

    function getKeeper() external view returns (address);

    function owner() external view returns (address);

    /* Setters */

    function setOtokenImpl(address _otokenImpl) external;

    function setOtokenFactory(address _factory) external;

    function setOracleImpl(address _otokenImpl) external;

    function setWhitelist(address _whitelist) external;

    function setController(address _controller) external;

    function setMarginPool(address _marginPool) external;

    function setMarginCalculator(address _calculator) external;

    function setControllerLogic(address _settlement) external;

    function setAddress(bytes32 _id, address _newImpl) external;
}


// ===== FILE: contracts/interfaces/OtokenInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

interface OtokenInterface {
    function addressBook() external view returns (address);

    function underlyingAsset() external view returns (address);

    function strikeAsset() external view returns (address);

    function collateralAsset() external view returns (address);

    function strikePrice() external view returns (uint256);

    function expiryTimestamp() external view returns (uint256);

    function isPut() external view returns (bool);

    function isPhysicallySettled() external view returns (bool);

    function init(
        address _addressBook,
        address _underlyingAsset,
        address _strikeAsset,
        address _collateralAsset,
        uint256 _strikePrice,
        uint256 _expiry,
        bool _isPut,
        bool _isPhysicallySettled
    ) external;

    function getOtokenDetails()
        external
        view
        returns (
            address,
            address,
            address,
            uint256,
            uint256,
            bool,
            bool
        );

    function mintOtoken(address account, uint256 amount) external;

    function burnOtoken(address account, uint256 amount) external;
}


// ===== FILE: contracts/interfaces/MarginCalculatorInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

pragma experimental ABIEncoderV2;

import {MarginVault} from "../libs/MarginVault.sol";

interface MarginCalculatorInterface {
    function addressBook() external view returns (address);

    function getExpiredPayoutRate(address _otoken) external view returns (uint256);

    function getStrikePaymentAmount(address _otoken, uint256 _amount) external view returns (uint256);

    function getExcessCollateral(MarginVault.Vault calldata _vault, uint256 _vaultType)
        external
        view
        returns (uint256 netValue, bool isExcess);

    function getFeeInformation() external view returns (uint256, address);
}


// ===== FILE: contracts/interfaces/OracleInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

interface OracleInterface {
    function isLockingPeriodOver(address _asset, uint256 _expiryTimestamp) external view returns (bool);

    function isDisputePeriodOver(address _asset, uint256 _expiryTimestamp) external view returns (bool);

    function getExpiryPrice(address _asset, uint256 _expiryTimestamp) external view returns (uint256, bool);

    function getDisputer() external view returns (address);

    function getPricer(address _asset) external view returns (address);

    function getPrice(address _asset) external view returns (uint256);

    function getPricerLockingPeriod(address _pricer) external view returns (uint256);

    function getPricerDisputePeriod(address _pricer) external view returns (uint256);

    function getChainlinkRoundData(address _asset, uint80 _roundId) external view returns (uint256, uint256);

    // Non-view function

    function setAssetPricer(address _asset, address _pricer) external;

    function setLockingPeriod(address _pricer, uint256 _lockingPeriod) external;

    function setDisputePeriod(address _pricer, uint256 _disputePeriod) external;

    function setExpiryPrice(
        address _asset,
        uint256 _expiryTimestamp,
        uint256 _price
    ) external;

    function disputeExpiryPrice(
        address _asset,
        uint256 _expiryTimestamp,
        uint256 _price
    ) external;

    function setDisputer(address _disputer) external;
}


// ===== FILE: contracts/interfaces/WhitelistInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

interface WhitelistInterface {
    /* View functions */

    function addressBook() external view returns (address);

    function isWhitelistedProduct(
        address _underlying,
        address _strike,
        address _collateral,
        bool _isPut
    ) external view returns (bool);

    function isWhitelistedCollateral(address _collateral) external view returns (bool);

    function isWhitelistedOtoken(address _otoken) external view returns (bool);

    /* Admin / factory only functions */
    function whitelistProduct(
        address _underlying,
        address _strike,
        address _collateral,
        bool _isPut
    ) external;

    function blacklistProduct(
        address _underlying,
        address _strike,
        address _collateral,
        bool _isPut
    ) external;

    function whitelistCollateral(address _collateral) external;

    function blacklistCollateral(address _collateral) external;

    function whitelistOtoken(address _otoken) external;

    function blacklistOtoken(address _otoken) external;
}


// ===== FILE: contracts/interfaces/MarginPoolInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;

interface MarginPoolInterface {
    /* Getters */
    function addressBook() external view returns (address);

    function farmer() external view returns (address);

    function getStoredBalance(address _asset) external view returns (uint256);

    /* Admin-only functions */
    function setFarmer(address _farmer) external;

    function farm(
        address _asset,
        address _receiver,
        uint256 _amount
    ) external;

    /* Controller-only functions */
    function transferToPool(
        address _asset,
        address _user,
        uint256 _amount
    ) external;

    function transferToUser(
        address _asset,
        address _user,
        uint256 _amount
    ) external;

    function decrementBalanceAfterBurn(address _otoken, uint256 _amount) external;

    function batchTransferToPool(
        address[] calldata _asset,
        address[] calldata _user,
        uint256[] calldata _amount
    ) external;

    function batchTransferToUser(
        address[] calldata _asset,
        address[] calldata _user,
        uint256[] calldata _amount
    ) external;

    function updateRedemptionBalance(
        address _otoken,
        int256 _otokenAmount,
        bool _isStrikeAsset,
        int256 _assetAmount
    ) external;

    function getRedemptionBalance(address _otoken)
        external
        view
        returns (
            uint256,
            uint256,
            uint256
        );
}


// ===== FILE: contracts/interfaces/ControllerInterface.sol =====
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.6.10;
pragma experimental ABIEncoderV2;

import {MarginVault} from "../libs/MarginVault.sol";

interface ControllerInterface {
    function getVaultWithDetails(address, uint256)
        external
        view
        returns (
            MarginVault.Vault memory,
            uint256,
            uint256
        );

    function canSettleAssets(
        address _underlying,
        address _strike,
        address _collateral,
        uint256 _expiry
    ) external view returns (bool);

    function removeVaultCollateral(
        address _owner,
        uint256 _vaultId,
        address _asset,
        uint256 _amount
    ) external;

    function updateVault(
        uint8 _action,
        address _owner,
        uint256 _vaultId,
        address _asset,
        uint256 _amount
    ) external;

    function deleteVault(address, uint256) external;

    function redeemTimePeriod() external view returns (uint256);
}


// ===== FILE: contracts/packages/oz/upgradeability/GSN/ContextUpgradeable.sol =====
// SPDX-License-Identifier: MIT
// openzeppelin-contracts-upgradeable v3.0.0

pragma solidity >=0.6.0 <0.8.0;

import "../Initializable.sol";

/*
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with GSN meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract ContextUpgradeable is Initializable {
    function __Context_init() internal initializer {
        __Context_init_unchained();
    }

    function __Context_init_unchained() internal initializer {}

    function _msgSender() internal view virtual returns (address payable) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }

    uint256[50] private __gap;
}
