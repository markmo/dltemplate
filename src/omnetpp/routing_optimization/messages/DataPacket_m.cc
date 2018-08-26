//
// Generated file, do not edit! Created by nedtool 4.6 from messages/DataPacket.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#include <iostream>
#include <sstream>
#include "DataPacket_m.h"

USING_NAMESPACE


// Another default rule (prevents compiler from choosing base class' doPacking())
template<typename T>
void doPacking(cCommBuffer *, T& t) {
    throw cRuntimeError("Parsim error: no doPacking() function for type %s or its base class (check .msg and _m.cc/h files!)",opp_typename(typeid(t)));
}

template<typename T>
void doUnpacking(cCommBuffer *, T& t) {
    throw cRuntimeError("Parsim error: no doUnpacking() function for type %s or its base class (check .msg and _m.cc/h files!)",opp_typename(typeid(t)));
}




// Template rule for outputting std::vector<T> types
template<typename T, typename A>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec)
{
    out.put('{');
    for(typename std::vector<T,A>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        if (it != vec.begin()) {
            out.put(','); out.put(' ');
        }
        out << *it;
    }
    out.put('}');
    
    char buf[32];
    sprintf(buf, " (size=%u)", (unsigned int)vec.size());
    out.write(buf, strlen(buf));
    return out;
}

// Template rule which fires if a struct or class doesn't have operator<<
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const T&) {return out;}

Register_Class(DataPacket);

DataPacket::DataPacket(const char *name, int kind) : ::cPacket(name,kind)
{
    this->srcNode_var = 0;
    this->dstNode_var = 0;
    this->ttl_var = 0;
    this->lastRouter_var = 0;
    this->l2_var = 0;
    this->l3_var = 0;
    this->l4_var = 0;
    this->lastQueue_var = 0;
    this->q2_var = 0;
    this->q3_var = 0;
    this->q4_var = 0;
    this->q5_var = 0;
    this->lastTS_var = 0;
    this->t2_var = 0;
    this->t3_var = 0;
    this->t4_var = 0;
    this->t5_var = 0;
    this->routing_var = 0;
}

DataPacket::DataPacket(const DataPacket& other) : ::cPacket(other)
{
    copy(other);
}

DataPacket::~DataPacket()
{
}

DataPacket& DataPacket::operator=(const DataPacket& other)
{
    if (this==&other) return *this;
    ::cPacket::operator=(other);
    copy(other);
    return *this;
}

void DataPacket::copy(const DataPacket& other)
{
    this->srcNode_var = other.srcNode_var;
    this->dstNode_var = other.dstNode_var;
    this->ttl_var = other.ttl_var;
    this->lastRouter_var = other.lastRouter_var;
    this->l2_var = other.l2_var;
    this->l3_var = other.l3_var;
    this->l4_var = other.l4_var;
    this->lastQueue_var = other.lastQueue_var;
    this->q2_var = other.q2_var;
    this->q3_var = other.q3_var;
    this->q4_var = other.q4_var;
    this->q5_var = other.q5_var;
    this->lastTS_var = other.lastTS_var;
    this->t2_var = other.t2_var;
    this->t3_var = other.t3_var;
    this->t4_var = other.t4_var;
    this->t5_var = other.t5_var;
    this->routing_var = other.routing_var;
}

void DataPacket::parsimPack(cCommBuffer *b)
{
    ::cPacket::parsimPack(b);
    doPacking(b,this->srcNode_var);
    doPacking(b,this->dstNode_var);
    doPacking(b,this->ttl_var);
    doPacking(b,this->lastRouter_var);
    doPacking(b,this->l2_var);
    doPacking(b,this->l3_var);
    doPacking(b,this->l4_var);
    doPacking(b,this->lastQueue_var);
    doPacking(b,this->q2_var);
    doPacking(b,this->q3_var);
    doPacking(b,this->q4_var);
    doPacking(b,this->q5_var);
    doPacking(b,this->lastTS_var);
    doPacking(b,this->t2_var);
    doPacking(b,this->t3_var);
    doPacking(b,this->t4_var);
    doPacking(b,this->t5_var);
    doPacking(b,this->routing_var);
}

void DataPacket::parsimUnpack(cCommBuffer *b)
{
    ::cPacket::parsimUnpack(b);
    doUnpacking(b,this->srcNode_var);
    doUnpacking(b,this->dstNode_var);
    doUnpacking(b,this->ttl_var);
    doUnpacking(b,this->lastRouter_var);
    doUnpacking(b,this->l2_var);
    doUnpacking(b,this->l3_var);
    doUnpacking(b,this->l4_var);
    doUnpacking(b,this->lastQueue_var);
    doUnpacking(b,this->q2_var);
    doUnpacking(b,this->q3_var);
    doUnpacking(b,this->q4_var);
    doUnpacking(b,this->q5_var);
    doUnpacking(b,this->lastTS_var);
    doUnpacking(b,this->t2_var);
    doUnpacking(b,this->t3_var);
    doUnpacking(b,this->t4_var);
    doUnpacking(b,this->t5_var);
    doUnpacking(b,this->routing_var);
}

int DataPacket::getSrcNode() const
{
    return srcNode_var;
}

void DataPacket::setSrcNode(int srcNode)
{
    this->srcNode_var = srcNode;
}

int DataPacket::getDstNode() const
{
    return dstNode_var;
}

void DataPacket::setDstNode(int dstNode)
{
    this->dstNode_var = dstNode;
}

int DataPacket::getTtl() const
{
    return ttl_var;
}

void DataPacket::setTtl(int ttl)
{
    this->ttl_var = ttl;
}

int DataPacket::getLastRouter() const
{
    return lastRouter_var;
}

void DataPacket::setLastRouter(int lastRouter)
{
    this->lastRouter_var = lastRouter;
}

int DataPacket::getL2() const
{
    return l2_var;
}

void DataPacket::setL2(int l2)
{
    this->l2_var = l2;
}

int DataPacket::getL3() const
{
    return l3_var;
}

void DataPacket::setL3(int l3)
{
    this->l3_var = l3;
}

int DataPacket::getL4() const
{
    return l4_var;
}

void DataPacket::setL4(int l4)
{
    this->l4_var = l4;
}

int DataPacket::getLastQueue() const
{
    return lastQueue_var;
}

void DataPacket::setLastQueue(int lastQueue)
{
    this->lastQueue_var = lastQueue;
}

int DataPacket::getQ2() const
{
    return q2_var;
}

void DataPacket::setQ2(int q2)
{
    this->q2_var = q2;
}

int DataPacket::getQ3() const
{
    return q3_var;
}

void DataPacket::setQ3(int q3)
{
    this->q3_var = q3;
}

int DataPacket::getQ4() const
{
    return q4_var;
}

void DataPacket::setQ4(int q4)
{
    this->q4_var = q4;
}

int DataPacket::getQ5() const
{
    return q5_var;
}

void DataPacket::setQ5(int q5)
{
    this->q5_var = q5;
}

double DataPacket::getLastTS() const
{
    return lastTS_var;
}

void DataPacket::setLastTS(double lastTS)
{
    this->lastTS_var = lastTS;
}

double DataPacket::getT2() const
{
    return t2_var;
}

void DataPacket::setT2(double t2)
{
    this->t2_var = t2;
}

double DataPacket::getT3() const
{
    return t3_var;
}

void DataPacket::setT3(double t3)
{
    this->t3_var = t3;
}

double DataPacket::getT4() const
{
    return t4_var;
}

void DataPacket::setT4(double t4)
{
    this->t4_var = t4;
}

double DataPacket::getT5() const
{
    return t5_var;
}

void DataPacket::setT5(double t5)
{
    this->t5_var = t5;
}

int DataPacket::getRouting() const
{
    return routing_var;
}

void DataPacket::setRouting(int routing)
{
    this->routing_var = routing;
}

class DataPacketDescriptor : public cClassDescriptor
{
  public:
    DataPacketDescriptor();
    virtual ~DataPacketDescriptor();

    virtual bool doesSupport(cObject *obj) const;
    virtual const char *getProperty(const char *propertyname) const;
    virtual int getFieldCount(void *object) const;
    virtual const char *getFieldName(void *object, int field) const;
    virtual int findField(void *object, const char *fieldName) const;
    virtual unsigned int getFieldTypeFlags(void *object, int field) const;
    virtual const char *getFieldTypeString(void *object, int field) const;
    virtual const char *getFieldProperty(void *object, int field, const char *propertyname) const;
    virtual int getArraySize(void *object, int field) const;

    virtual std::string getFieldAsString(void *object, int field, int i) const;
    virtual bool setFieldAsString(void *object, int field, int i, const char *value) const;

    virtual const char *getFieldStructName(void *object, int field) const;
    virtual void *getFieldStructPointer(void *object, int field, int i) const;
};

Register_ClassDescriptor(DataPacketDescriptor);

DataPacketDescriptor::DataPacketDescriptor() : cClassDescriptor("DataPacket", "cPacket")
{
}

DataPacketDescriptor::~DataPacketDescriptor()
{
}

bool DataPacketDescriptor::doesSupport(cObject *obj) const
{
    return dynamic_cast<DataPacket *>(obj)!=NULL;
}

const char *DataPacketDescriptor::getProperty(const char *propertyname) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : NULL;
}

int DataPacketDescriptor::getFieldCount(void *object) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 18+basedesc->getFieldCount(object) : 18;
}

unsigned int DataPacketDescriptor::getFieldTypeFlags(void *object, int field) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldTypeFlags(object, field);
        field -= basedesc->getFieldCount(object);
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
    };
    return (field>=0 && field<18) ? fieldTypeFlags[field] : 0;
}

const char *DataPacketDescriptor::getFieldName(void *object, int field) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldName(object, field);
        field -= basedesc->getFieldCount(object);
    }
    static const char *fieldNames[] = {
        "srcNode",
        "dstNode",
        "ttl",
        "lastRouter",
        "l2",
        "l3",
        "l4",
        "lastQueue",
        "q2",
        "q3",
        "q4",
        "q5",
        "lastTS",
        "t2",
        "t3",
        "t4",
        "t5",
        "routing",
    };
    return (field>=0 && field<18) ? fieldNames[field] : NULL;
}

int DataPacketDescriptor::findField(void *object, const char *fieldName) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount(object) : 0;
    if (fieldName[0]=='s' && strcmp(fieldName, "srcNode")==0) return base+0;
    if (fieldName[0]=='d' && strcmp(fieldName, "dstNode")==0) return base+1;
    if (fieldName[0]=='t' && strcmp(fieldName, "ttl")==0) return base+2;
    if (fieldName[0]=='l' && strcmp(fieldName, "lastRouter")==0) return base+3;
    if (fieldName[0]=='l' && strcmp(fieldName, "l2")==0) return base+4;
    if (fieldName[0]=='l' && strcmp(fieldName, "l3")==0) return base+5;
    if (fieldName[0]=='l' && strcmp(fieldName, "l4")==0) return base+6;
    if (fieldName[0]=='l' && strcmp(fieldName, "lastQueue")==0) return base+7;
    if (fieldName[0]=='q' && strcmp(fieldName, "q2")==0) return base+8;
    if (fieldName[0]=='q' && strcmp(fieldName, "q3")==0) return base+9;
    if (fieldName[0]=='q' && strcmp(fieldName, "q4")==0) return base+10;
    if (fieldName[0]=='q' && strcmp(fieldName, "q5")==0) return base+11;
    if (fieldName[0]=='l' && strcmp(fieldName, "lastTS")==0) return base+12;
    if (fieldName[0]=='t' && strcmp(fieldName, "t2")==0) return base+13;
    if (fieldName[0]=='t' && strcmp(fieldName, "t3")==0) return base+14;
    if (fieldName[0]=='t' && strcmp(fieldName, "t4")==0) return base+15;
    if (fieldName[0]=='t' && strcmp(fieldName, "t5")==0) return base+16;
    if (fieldName[0]=='r' && strcmp(fieldName, "routing")==0) return base+17;
    return basedesc ? basedesc->findField(object, fieldName) : -1;
}

const char *DataPacketDescriptor::getFieldTypeString(void *object, int field) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldTypeString(object, field);
        field -= basedesc->getFieldCount(object);
    }
    static const char *fieldTypeStrings[] = {
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "double",
        "double",
        "double",
        "double",
        "double",
        "int",
    };
    return (field>=0 && field<18) ? fieldTypeStrings[field] : NULL;
}

const char *DataPacketDescriptor::getFieldProperty(void *object, int field, const char *propertyname) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldProperty(object, field, propertyname);
        field -= basedesc->getFieldCount(object);
    }
    switch (field) {
        default: return NULL;
    }
}

int DataPacketDescriptor::getArraySize(void *object, int field) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getArraySize(object, field);
        field -= basedesc->getFieldCount(object);
    }
    DataPacket *pp = (DataPacket *)object; (void)pp;
    switch (field) {
        default: return 0;
    }
}

std::string DataPacketDescriptor::getFieldAsString(void *object, int field, int i) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldAsString(object,field,i);
        field -= basedesc->getFieldCount(object);
    }
    DataPacket *pp = (DataPacket *)object; (void)pp;
    switch (field) {
        case 0: return long2string(pp->getSrcNode());
        case 1: return long2string(pp->getDstNode());
        case 2: return long2string(pp->getTtl());
        case 3: return long2string(pp->getLastRouter());
        case 4: return long2string(pp->getL2());
        case 5: return long2string(pp->getL3());
        case 6: return long2string(pp->getL4());
        case 7: return long2string(pp->getLastQueue());
        case 8: return long2string(pp->getQ2());
        case 9: return long2string(pp->getQ3());
        case 10: return long2string(pp->getQ4());
        case 11: return long2string(pp->getQ5());
        case 12: return double2string(pp->getLastTS());
        case 13: return double2string(pp->getT2());
        case 14: return double2string(pp->getT3());
        case 15: return double2string(pp->getT4());
        case 16: return double2string(pp->getT5());
        case 17: return long2string(pp->getRouting());
        default: return "";
    }
}

bool DataPacketDescriptor::setFieldAsString(void *object, int field, int i, const char *value) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->setFieldAsString(object,field,i,value);
        field -= basedesc->getFieldCount(object);
    }
    DataPacket *pp = (DataPacket *)object; (void)pp;
    switch (field) {
        case 0: pp->setSrcNode(string2long(value)); return true;
        case 1: pp->setDstNode(string2long(value)); return true;
        case 2: pp->setTtl(string2long(value)); return true;
        case 3: pp->setLastRouter(string2long(value)); return true;
        case 4: pp->setL2(string2long(value)); return true;
        case 5: pp->setL3(string2long(value)); return true;
        case 6: pp->setL4(string2long(value)); return true;
        case 7: pp->setLastQueue(string2long(value)); return true;
        case 8: pp->setQ2(string2long(value)); return true;
        case 9: pp->setQ3(string2long(value)); return true;
        case 10: pp->setQ4(string2long(value)); return true;
        case 11: pp->setQ5(string2long(value)); return true;
        case 12: pp->setLastTS(string2double(value)); return true;
        case 13: pp->setT2(string2double(value)); return true;
        case 14: pp->setT3(string2double(value)); return true;
        case 15: pp->setT4(string2double(value)); return true;
        case 16: pp->setT5(string2double(value)); return true;
        case 17: pp->setRouting(string2long(value)); return true;
        default: return false;
    }
}

const char *DataPacketDescriptor::getFieldStructName(void *object, int field) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldStructName(object, field);
        field -= basedesc->getFieldCount(object);
    }
    switch (field) {
        default: return NULL;
    };
}

void *DataPacketDescriptor::getFieldStructPointer(void *object, int field, int i) const
{
    cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount(object))
            return basedesc->getFieldStructPointer(object, field, i);
        field -= basedesc->getFieldCount(object);
    }
    DataPacket *pp = (DataPacket *)object; (void)pp;
    switch (field) {
        default: return NULL;
    }
}


