//----------------------------------------------------------------------------
/** @file GtpOutputStream.h */
//----------------------------------------------------------------------------

#ifndef GTP_OUTPUTSTREAM_H
#define GTP_OUTPUTSTREAM_H

#include <iostream>
#include <string>

//----------------------------------------------------------------------------

/** Base class for output streams used by GtpEngine.
    This implementation only forwards calls to std::ostream.
    @todo Why does it need this class if users can write their own streams
    compatible with the standard library? See also
    https://sourceforge.net/apps/trac/fuego/ticket/66 */
class GtpOutputStream
{
public:
    GtpOutputStream(std::ostream &out);

    virtual ~GtpOutputStream();

    virtual void Write(const std::string &line);

    virtual void Flush();

private:
    std::ostream &m_out;
};

//----------------------------------------------------------------------------

#endif // GTP_OUTPUTSTREAM_H

